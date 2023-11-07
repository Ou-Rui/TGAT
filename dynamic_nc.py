""" 
用DDGCL的训练方式做动态节点分类
python -u dynamic_nc.py
"""
import math
import logging
import time
import sys
import random
import argparse

from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from module import TGAN
from graph import NeighborFinder


class LR(torch.nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=True)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


random.seed(222)
np.random.seed(222)
torch.manual_seed(222)

### Argument and global variables
parser = argparse.ArgumentParser('Dynamic Node Classification')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=100, help='batch_size')
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--tune', action='store_true',
                    help='parameters tuning mode, use train-test split on training data only.')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method',
                    default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information',
                    default='time')

parser.add_argument('--new_node', action='store_true', help='model new node')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NEW_NODE = args.new_node
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 2      # 2层
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Load data and train val test split
g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))
e_feat = np.load('./processed/ml_{}.npy'.format(DATA))
n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA))

# 以下5个数据结构：ndarray: [E]。排序：ts_l递增
src_l = g_df.u.values       # 节点编号从1开始
dst_l = g_df.i.values
e_idx_l = g_df.idx.values   # eid已经按照时间顺序组织，编号从1开始，1,2,3,4...
label_l = g_df.label.values
ts_l = g_df.ts.values

# 按照时间划分数据集，0.7 : 0.15 : 0.15
val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))
valid_train_flag = (ts_l <= val_time)
valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
valid_test_flag = ts_l > test_time



max_src_index = src_l.max()     # no use
max_idx = max(src_l.max(), dst_l.max())     # number of nodes

total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))

# Training set
train_src_l = src_l[valid_train_flag]
train_dst_l = dst_l[valid_train_flag]
train_ts_l = ts_l[valid_train_flag]
train_e_idx_l = e_idx_l[valid_train_flag]
train_label_l = label_l[valid_train_flag]
# Validation set
val_src_l = src_l[valid_val_flag]
val_dst_l = dst_l[valid_val_flag]
val_ts_l = ts_l[valid_val_flag]
val_e_idx_l = e_idx_l[valid_val_flag]
val_label_l = label_l[valid_val_flag]
# Test set
test_src_l = src_l[valid_test_flag]
test_dst_l = dst_l[valid_test_flag]
test_ts_l = ts_l[valid_test_flag]
test_e_idx_l = e_idx_l[valid_test_flag]
test_label_l = label_l[valid_test_flag]

### Initialize the data structure for graph and edge sampling
# 无向图的邻接表，存三元组(dst_nid, eid, ts)
adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)

# full graph with all the data for the test and validation purpose
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)

### Model initialize
device = torch.device('cuda:{}'.format(GPU))
tgan = TGAN(train_ngh_finder, n_feat, e_feat, num_layers=NUM_LAYER,
            attn_mode=ATTN_MODE, use_time=USE_TIME, agg_method=AGG_METHOD,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT,
            node_dim=NODE_DIM, time_dim=TIME_DIM)   # node_dim & time_dim no use
# optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
# criterion = torch.nn.BCELoss()
tgan = tgan.to(device)

num_instance = len(train_src_l)     # 训练集边的数量
num_batch = math.ceil(num_instance / BATCH_SIZE)
logger.debug(f'num of training instances: {num_instance}')
logger.debug(f'num of batches per epoch: {num_batch}')
# idx_list = np.arange(num_instance)

# logger.info('loading saved TGAN model')
# model_path = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{DATA}.pth'
# tgan.load_state_dict(torch.load(model_path))
# tgan.eval()
# logger.info('TGAN model loaded')
# logger.info('Start training node classification task')

""" NC Decoder: 3-layer MLP """
lr_model = LR(n_feat.shape[1])
lr_optimizer = torch.optim.Adam(lr_model.parameters(), lr=args.lr)
lr_model = lr_model.to(device)
# tgan.ngh_finder = full_ngh_finder     # 为啥原来是用full_ngh_finder??
# idx_list = np.arange(num_instance)
# criterion为什么还分两个？
lr_criterion = torch.nn.BCELoss()
lr_criterion_eval = torch.nn.BCELoss()


def eval_epoch(src_l, dst_l, ts_l, label_l, batch_size, lr_model, tgan, num_layer=NODE_LAYER):
    pred_prob = np.zeros(len(src_l))
    loss = 0
    num_instance = len(src_l)
    num_batch = math.ceil(num_instance / batch_size)
    with torch.no_grad():
        lr_model.eval()
        tgan.eval()
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance - 1, s_idx + batch_size)
            src_l_cut = src_l[s_idx:e_idx]
            dst_l_cut = dst_l[s_idx:e_idx]
            ts_l_cut = ts_l[s_idx:e_idx]
            label_l_cut = label_l[s_idx:e_idx]
            size = len(src_l_cut)
            ''' NC调用tem_conv计算节点表示 '''
            src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, num_layer)
            src_label = torch.from_numpy(label_l_cut).float().to(device)
            lr_prob = lr_model(src_embed).sigmoid()
            loss += lr_criterion_eval(lr_prob, src_label).item()
            pred_prob[s_idx:e_idx] = lr_prob.cpu().numpy()

    auc_roc = roc_auc_score(label_l, pred_prob)
    return auc_roc, loss / num_instance


for epoch in tqdm(range(args.n_epoch)):
    lr_pred_prob = np.zeros(len(train_src_l))
    # Train encoder & decoder
    tgan = tgan.train()              
    lr_model = lr_model.train()
    tgan.ngh_finder = train_ngh_finder
    # batch loop
    for k in range(num_batch):
        s_idx = k * BATCH_SIZE
        e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
        src_l_cut = train_src_l[s_idx:e_idx]
        dst_l_cut = train_dst_l[s_idx:e_idx]
        ts_l_cut = train_ts_l[s_idx:e_idx]
        label_l_cut = train_label_l[s_idx:e_idx]

        lr_optimizer.zero_grad()
        with torch.no_grad():
            src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, NODE_LAYER)

        src_label = torch.from_numpy(label_l_cut).float().to(device)
        lr_prob = lr_model(src_embed).sigmoid()
        lr_loss = lr_criterion(lr_prob, src_label)
        lr_loss.backward()
        lr_optimizer.step()

    tgan.ngh_finder = full_ngh_finder
    train_auc, train_loss = eval_epoch(train_src_l, train_dst_l, train_ts_l, train_label_l,
                                       BATCH_SIZE, lr_model, tgan)
    val_auc, val_loss = eval_epoch(val_src_l, val_dst_l, val_ts_l, val_label_l,
                                   BATCH_SIZE, lr_model, tgan)
    test_auc, test_loss = eval_epoch(test_src_l, test_dst_l, test_ts_l, test_label_l,
                                     BATCH_SIZE, lr_model, tgan)
    # torch.save(lr_model.state_dict(), './saved_models/dnc_{}_wiki_node_class.pth'.format(DATA))
    logger.info(f'train auc: {train_auc:.4f}, val auc: {val_auc:.4f}, test auc: {test_auc:.4f}')

test_auc, test_loss = eval_epoch(test_src_l, test_dst_l, test_ts_l, test_label_l,
                                 BATCH_SIZE, lr_model, tgan)
# torch.save(lr_model.state_dict(), './saved_models/dnc_{}_decoder.pth'.format(DATA))
logger.info(f'test auc: {test_auc}')
