"""
Unified interface to all dynamic graph model experiments

python -u learn_node_c.py -d wikipedia --bs 100 --uniform --prefix 240327
python -u learn_node_c.py -d reddit --bs 100 --uniform --prefix 240327
python -u learn_node_c.py -d mooc --bs 100 --uniform --prefix 240327
python -u learn_node_c.py -d txn_filter --bs 100 --uniform --prefix 240325

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
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, f1_score, average_precision_score

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
    h1 = x
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    h2 = x
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1), h1, h2


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
parser.add_argument('--n_run', type=int, default=10, help='number of runs')
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

if args.data in ('txn', 'txn_filter'):
  val_time, test_time = list(np.quantile(g_df.ts, [0.34, 0.67]))
elif args.data in ('wikipedia', 'reddit', 'mooc'):
  val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))
else:
  raise ValueError("invalid dataset")

# 以下5个数据结构：ndarray: [E]。排序：ts_l递增
src_l = g_df.u.values       # 节点编号从1开始
dst_l = g_df.i.values
e_idx_l = g_df.idx.values   # eid已经按照时间顺序组织，编号从1开始，1,2,3,4...
label_l = g_df.label.values
ts_l = g_df.ts.values

max_src_index = src_l.max()     # no use
max_idx = max(src_l.max(), dst_l.max())     # number of nodes

total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))

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


def eval_epoch(src_l, dst_l, ts_l, label_l, batch_size, lr_model, tgan, num_layer=NODE_LAYER):
  pred_prob = np.zeros(len(src_l))
  loss = 0
  num_instance = len(src_l)
  num_batch = math.ceil(num_instance / batch_size)
  emb_l, h1_l, h2_l = [], [], []
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
      lr_prob, h1, h2 = lr_model(src_embed)
      lr_prob = lr_prob.sigmoid()
      loss += lr_criterion_eval(lr_prob, src_label).item()
      pred_prob[s_idx:e_idx] = lr_prob.cpu().numpy()
      
      emb_l.extend(src_embed.cpu().numpy())
      h1_l.extend(h1.cpu().numpy())
      h2_l.extend(h2.cpu().numpy())

  fpr, tpr, thresholds = roc_curve(label_l, pred_prob)
  # auc_score = auc(fpr, tpr)
  # 找到最佳阈值对应的索引
  optimal_idx = np.argmax(tpr - fpr)
  optimal_threshold = thresholds[optimal_idx]
  pred_labels = pred_prob > optimal_threshold
  f1 = f1_score(label_l, pred_labels)
  recall = recall_score(label_l, pred_labels)
  auc_roc = roc_auc_score(label_l, pred_prob)
  ap = average_precision_score(label_l, pred_prob)
  metrics = (auc_roc, ap, f1, recall)
  return metrics, loss / num_instance, emb_l, pred_prob, h1_l, h2_l

def save_embs(epoch, emb_tuple, prob_tuple, h1_tuple, h2_tuple):
  emb_l, prob_l, h1_l, h2_l = [], [], [], []
  for emb_mode, prob_mode, h1_mode, h2_mode in zip(emb_tuple, prob_tuple, h1_tuple, h2_tuple):
    emb_l.extend(emb_mode)
    prob_l.extend(prob_mode)
    h1_l.extend(h1_mode)
    h2_l.extend(h2_mode)
  emb_l = np.array(emb_l)
  prob_l = np.array(prob_l)
  h1_l = np.array(h1_l)
  h2_l = np.array(h2_l)
  np.save(f"./saved_embs/TGAT_{args.prefix}_{args.data}_epoch{epoch}_embs.npy", emb_l)
  np.save(f"./saved_embs/TGAT_{args.prefix}_{args.data}_epoch{epoch}_probs.npy", prob_l)
  np.save(f"./saved_embs/TGAT_{args.prefix}_{args.data}_epoch{epoch}_h1.npy", h1_l)
  np.save(f"./saved_embs/TGAT_{args.prefix}_{args.data}_epoch{epoch}_h2.npy", h2_l)


''' Train Procedure'''
METRICS = ['AUC', 'AP', 'F1', 'RECALL']
best_epoch_l = []
best_val_metric_l = []
best_test_metric_l = []
max_test_metric_l = []
for i_run in range(args.n_run):
  logger.info(f'\n =========================== RUN {i_run} START ==================================')
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
  
  best_epoch = [0, 0, 0, 0]
  best_val_metric = [0, 0, 0, 0]
  best_test_metric = [0, 0, 0, 0]
  max_test_metric = [0, 0, 0, 0]
  # Epoch Loop
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
      lr_prob, h1, h2 = lr_model(src_embed)
      lr_prob = lr_prob.sigmoid()
      lr_loss = lr_criterion(lr_prob, src_label)
      lr_loss.backward()
      lr_optimizer.step()
    # End Batch Loop
    
    train_metric, train_loss, train_emb_l, train_prob_l, train_h1_l, train_h2_l = \
        eval_epoch(train_src_l, train_dst_l, train_ts_l, train_label_l,
                  BATCH_SIZE, lr_model, tgan)
    val_metric, val_loss, val_emb_l, val_prob_l, val_h1_l, val_h2_l = \
        eval_epoch(val_src_l, val_dst_l, val_ts_l, val_label_l,
                  BATCH_SIZE, lr_model, tgan)
    test_metric, test_loss, test_emb_l, test_prob_l, test_h1_l, test_h2_l = \
        eval_epoch(test_src_l, test_dst_l, test_ts_l, test_label_l,
                  BATCH_SIZE, lr_model, tgan)
        
    train_auc, train_ap, train_f1, train_recall = train_metric
    val_auc, val_ap, val_f1, val_recall = val_metric
    test_auc, test_ap, test_f1, test_recall = test_metric
    
    # torch.save(lr_model.state_dict(), './saved_models/edge_{}_wiki_node_class.pth'.format(DATA))
    logger.info(f'train_auc: {train_auc:.4f}, train_ap: {train_ap:.4f}, train_f1: {train_f1:.4f}, train_recall: {train_recall:.4f}')
    logger.info(f'val_auc: {val_auc:.4f}, val_ap: {val_ap:.4f}, val_f1: {val_f1:.4f}, val_recall: {val_recall:.4f}')
    logger.info(f'test_auc: {test_auc:.4f}, test_ap: {test_ap:.4f}, test_f1: {test_f1:.4f}, test_recall: {test_recall:.4f}')
    save_embs(epoch, (train_emb_l, val_emb_l, test_emb_l), (train_prob_l, val_prob_l, test_prob_l),
              (train_h1_l, val_h1_l, test_h1_l), (train_h2_l, val_h2_l, test_h2_l))
    
    for i in range(len(METRICS)):
      if val_metric[i] > best_val_metric[i]:
        best_val_metric[i] = val_metric[i]
        best_epoch[i] = epoch
        best_test_metric[i] = test_metric[i]
      if test_metric[i] > max_test_metric[i]:
        max_test_metric[i] = test_metric[i]
    
  # End Epoch Loop

  logger.info(f'\n =========================== RUN {i_run} END ==================================')
  for i in range(len(METRICS)):
    logger.debug(f'{METRICS[i]}: best_epoch={best_epoch[i]}, val_{METRICS[i]}={best_val_metric[i]:.4f}, ' + \
                  f'test_{METRICS[i]}={best_test_metric[i]:.4f}, max_test_{METRICS[i]}={max_test_metric[i]:.4f}')
  best_epoch_l.append(best_epoch)
  best_val_metric_l.append(best_val_metric)
  best_test_metric_l.append(best_test_metric)
  max_test_metric_l.append(max_test_metric)
# End Run Loop

logger.debug(f'========================== SUMMARY ==========================')

for i in range(len(METRICS)):
  logger.debug(f'<<< {METRICS[i]} >>>')
  for i_run in range(args.n_run):
    logger.debug(f'RUN #{i_run}: best_epoch={best_epoch_l[i_run][i]}, best_val_{METRICS[i]}={best_val_metric_l[i_run][i]:.4f}, ' + \
                 f'best_test_{best_test_metric_l[i_run][i]:.4f}, max_test_{METRICS[i]}={max_test_metric_l[i_run][i]:.4f}')
  
  logger.debug(f'ALL IN ALL -- ave_best_epoch: {round(sum(best_epoch_l[:][i])/args.n_run, 4)}')
  logger.debug(f'ALL IN ALL -- ave_best_val_{METRICS[i]}: {round(sum([x[i] for x in best_val_metric_l])/args.n_run, 4)}')
  logger.debug(f'ALL IN ALL -- ave_best_test_{METRICS[i]}: {round(sum([x[i] for x in best_test_metric_l])/args.n_run, 4)}' + \
              f' \u00B1 {round(np.std([x[i] for x in best_test_metric_l]), 4)}')
  logger.debug(f'ALL IN ALL -- ave_max_test_{METRICS[i]}: {round(sum([x[i] for x in max_test_metric_l])/args.n_run, 4)}' + \
              f' \u00B1 {round(np.std([x[i] for x in max_test_metric_l]), 4)}')

# logger.info(f'\n =========================== SUMMARY ==================================')
# for i in range(len(best_metric_l)):
#   best_metric = best_metric_l[i]
#   logger.info(f'RUN #{i} -- auc: {best_metric[0]}, ap: {best_metric[1]}, f1: {best_metric[2]}, ' + \
#                f'recall: {best_metric[3]}')

# logger.info(f'ALL IN ALL -- ave_auc: {round(sum([x[0] for x in best_metric_l])/ args.n_run, 4)}' + \
#              f' \u00B1 {round(np.std([x[0] for x in best_metric_l]), 4)}')
# logger.info(f'ALL IN ALL -- ave_ap: {round(sum([x[1] for x in best_metric_l])/ args.n_run, 4)}' + \
#              f' \u00B1 {round(np.std([x[1] for x in best_metric_l]), 4)}')
# logger.info(f'ALL IN ALL -- ave_f1: {round(sum([x[2] for x in best_metric_l])/ args.n_run, 4)}' + \
#              f' \u00B1 {round(np.std([x[2] for x in best_metric_l]), 4)}')
# logger.info(f'ALL IN ALL -- ave_recall: {round(sum([x[3] for x in best_metric_l])/ args.n_run, 4)}' + \
#              f' \u00B1 {round(np.std([x[3] for x in best_metric_l]), 4)}')



