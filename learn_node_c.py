"""
Unified interface to all dynamic graph model experiments

python -u learn_node_c.py -d wikipedia --bs 100 --uniform --prefix 240321
python -u learn_node_c.py -d txn_filter --bs 100 --uniform --prefix 240323

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
    hidden_emb = x
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1), hidden_emb


random.seed(222)
np.random.seed(222)
torch.manual_seed(222)

### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on node classification')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=30, help='batch_size')
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--n_epoch', type=int, default=15, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--tune', action='store_true',
                    help='parameters tuning mode, use train-test split on training data only.')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=None, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=None, help='Dimensions of the time embedding')
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
NODE_LAYER = 1
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
e_feat = np.load('./processed/ml_{}.npy'.format(DATA))        # [E, 172]
n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA))   # [N, 172]
logger.info(f'e_feat.shape={e_feat.shape}, n_feat.shape={n_feat.shape}')

# 按照时间划分数据集，0.7 : 0.15 : 0.15
val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

if args.data == 'txn_filter':
  val_time, test_time = list(np.quantile(g_df.ts, [0.34, 0.66]))

# 以下5个数据结构：ndarray: [E]。排序：ts_l递增
src_l = g_df.u.values       # 节点编号从1开始
dst_l = g_df.i.values
e_idx_l = g_df.idx.values   # eid已经按照时间顺序组织，编号从1开始，1,2,3,4...
label_l = g_df.label.values
ts_l = g_df.ts.values

max_src_index = src_l.max()     # no use
max_idx = max(src_l.max(), dst_l.max())     # number of nodes

total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))

''' 前85%: train + val (8:2的比例, 随机分); 后15%: test'''
# valid_train_flag = (ts_l <= test_time)
# valid_val_flag = (ts_l <= test_time)
# # 在前85%时间的边中继续划分：80% train，20% val（随机划分）
# assignment = np.random.randint(0, 10, len(valid_train_flag))
# valid_train_flag *= (assignment >= 2)
# valid_val_flag *= (assignment < 2)
# valid_test_flag = ts_l > test_time

# if args.tune:
#     # 调参数模式，用val选最佳参数
#     train_src_l = src_l[valid_train_flag]
#     train_dst_l = dst_l[valid_train_flag]
#     train_ts_l = ts_l[valid_train_flag]
#     train_e_idx_l = e_idx_l[valid_train_flag]
#     train_label_l = label_l[valid_train_flag]
#     # use the validation as test dataset
#     test_src_l = src_l[valid_val_flag]
#     test_dst_l = dst_l[valid_val_flag]
#     test_ts_l = ts_l[valid_val_flag]
#     test_e_idx_l = e_idx_l[valid_val_flag]
#     test_label_l = label_l[valid_val_flag]
# else:
#     logger.info('Training use all train data')
#     valid_train_flag = (ts_l <= test_time)
#     train_src_l = src_l[valid_train_flag]
#     train_dst_l = dst_l[valid_train_flag]
#     train_ts_l = ts_l[valid_train_flag]
#     train_e_idx_l = e_idx_l[valid_train_flag]
#     train_label_l = label_l[valid_train_flag]
#     # use the true test dataset
#     test_src_l = src_l[valid_test_flag]
#     test_dst_l = dst_l[valid_test_flag]
#     test_ts_l = ts_l[valid_test_flag]
#     test_e_idx_l = e_idx_l[valid_test_flag]
#     test_label_l = label_l[valid_test_flag]

''' 正常划分 '''
valid_train_flag = (ts_l <= val_time)
valid_val_flag = (ts_l > val_time) * (ts_l <= test_time) 
valid_test_flag = ts_l > test_time
train_src_l = src_l[valid_train_flag]
train_dst_l = dst_l[valid_train_flag]
train_ts_l = ts_l[valid_train_flag]
train_e_idx_l = e_idx_l[valid_train_flag]
train_label_l = label_l[valid_train_flag]
val_src_l = src_l[valid_val_flag]
val_dst_l = dst_l[valid_val_flag]
val_ts_l = ts_l[valid_val_flag]
val_e_idx_l = e_idx_l[valid_val_flag]
val_label_l = label_l[valid_val_flag]
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

logger.info('loading saved TGAN model')
model_path = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{DATA}.pth'
tgan.load_state_dict(torch.load(model_path))
tgan.eval()
logger.info('TGAN model loaded')
logger.info('Start training node classification task')

""" NC Decoder: 3-layer MLP """
lr_model = LR(n_feat.shape[1])
lr_optimizer = torch.optim.Adam(lr_model.parameters(), lr=args.lr)
lr_model = lr_model.to(device)
tgan.ngh_finder = full_ngh_finder
# idx_list = np.arange(num_instance)
# criterion为什么还分两个？
lr_criterion = torch.nn.BCELoss()
lr_criterion_eval = torch.nn.BCELoss()


def eval_epoch(src_l, dst_l, ts_l, label_l, batch_size, lr_model, tgan, num_layer=NODE_LAYER):
  pred_prob = np.zeros(len(src_l))
  loss = 0
  num_instance = len(src_l)
  num_batch = math.ceil(num_instance / batch_size)
  emb_l, hidden_emb_l = [], []
  with torch.no_grad():
    lr_model.eval()
    tgan.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)
      src_l_cut = src_l[s_idx:e_idx]
      dst_l_cut = dst_l[s_idx:e_idx]
      ts_l_cut = ts_l[s_idx:e_idx]
      label_l_cut = label_l[s_idx:e_idx]
      size = len(src_l_cut)
      ''' NC调用tem_conv计算节点表示 '''
      src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, num_layer) # [b, 172]
      src_label = torch.from_numpy(label_l_cut).float().to(device)
      lr_prob, hidden_emb = lr_model(src_embed)
      lr_prob = lr_prob.sigmoid()
      loss += lr_criterion_eval(lr_prob, src_label).item()
      pred_prob[s_idx:e_idx] = lr_prob.cpu().numpy()
      
      emb_l.extend(src_embed.cpu().numpy())
      hidden_emb_l.extend(hidden_emb.cpu().numpy())

  auc_roc = roc_auc_score(label_l, pred_prob)
  return auc_roc, loss / num_instance, emb_l, pred_prob, hidden_emb_l


for epoch in tqdm(range(args.n_epoch)):
  lr_pred_prob = np.zeros(len(train_src_l))
  # np.random.shuffle(idx_list)         # why shuffle? no use
  tgan = tgan.eval()              # TGAT param fixed
  lr_model = lr_model.train()     # Train Decoder only
  # batch loop
  for k in range(num_batch):
    s_idx = k * BATCH_SIZE
    e_idx = min(num_instance, s_idx + BATCH_SIZE)
    src_l_cut = train_src_l[s_idx:e_idx]
    dst_l_cut = train_dst_l[s_idx:e_idx]
    ts_l_cut = train_ts_l[s_idx:e_idx]
    label_l_cut = train_label_l[s_idx:e_idx]

    lr_optimizer.zero_grad()
    with torch.no_grad():
      src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, NODE_LAYER)

    src_label = torch.from_numpy(label_l_cut).float().to(device)
    lr_prob, hidden_emb = lr_model(src_embed)
    lr_prob = lr_prob.sigmoid()
    lr_loss = lr_criterion(lr_prob, src_label)
    lr_loss.backward()
    lr_optimizer.step()

  train_auc, train_loss, _, _, _ = eval_epoch(train_src_l, train_dst_l, train_ts_l, train_label_l,
                                      BATCH_SIZE, lr_model, tgan)
  test_auc, test_loss, _, _, _ = eval_epoch(test_src_l, test_dst_l, test_ts_l, test_label_l,
                                    BATCH_SIZE, lr_model, tgan)
  # torch.save(lr_model.state_dict(), './saved_models/edge_{}_wiki_node_class.pth'.format(DATA))
  logger.info(f'train auc: {train_auc}, test auc: {test_auc}')

emb_l, hidden_emb_l, prob_l = [], [], []
train_auc, train_loss, train_emb, train_prob, train_hidden_emb = eval_epoch(train_src_l, train_dst_l, train_ts_l, train_label_l,
                                           BATCH_SIZE, lr_model, tgan)
val_auc, val_loss, val_emb, val_prob, val_hidden_emb = eval_epoch(val_src_l, val_dst_l, val_ts_l, val_label_l,
                                           BATCH_SIZE, lr_model, tgan)
test_auc, test_loss, test_emb, test_prob, test_hidden_emb = eval_epoch(test_src_l, test_dst_l, test_ts_l, test_label_l,
                                           BATCH_SIZE, lr_model, tgan)
emb_l.extend(train_emb)
emb_l.extend(val_emb)
emb_l.extend(test_emb)
hidden_emb_l.extend(train_hidden_emb)
hidden_emb_l.extend(val_hidden_emb)
hidden_emb_l.extend(test_hidden_emb)
prob_l.extend(train_prob)
prob_l.extend(val_prob)
prob_l.extend(test_prob)
# torch.save(lr_model.state_dict(), './saved_models/edge_{}_wiki_node_class.pth'.format(DATA))
logger.info(f'test auc: {test_auc}')
emb_dim = 128 if args.data == 'txn_filter' else 172
emb_l = np.array(emb_l).reshape([-1, emb_dim])
hidden_emb_l = np.array(hidden_emb_l)
assert hidden_emb_l.shape[0] == emb_l.shape[0]
prob_l = np.array(prob_l).reshape([-1, 1])
neg_prob_l = 1 - prob_l
prob_l = np.concatenate((neg_prob_l, prob_l), axis=1)
assert prob_l.shape[0] == emb_l.shape[0]
np.save(f"./saved_embs/TGAT_{args.prefix}_{args.data}_embs.npy", emb_l)
np.save(f"./saved_embs/TGAT_{args.prefix}_{args.data}_hidden2_embs.npy", hidden_emb_l)
np.save(f"./saved_embs/TGAT_{args.prefix}_{args.data}_probs.npy", prob_l)
