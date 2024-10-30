"""
持续学习实验
python learn_cont.py -d wikipedia
"""
import math
import logging
import time
import sys
import random
import argparse
from pathlib import Path

from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, f1_score, average_precision_score

from module import TGAN
from graph import NeighborFinder
from utils import get_ftime
from dataloader.get_data import get_continual_data

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


# random.seed(222)
# np.random.seed(222)
# torch.manual_seed(222)

### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on node classification')
parser.add_argument('-d', '--dataset', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
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
parser.add_argument('--mask', type=float, default=0.0, help='mask ratio of training data')

parser.add_argument('--num-task', type=int, default=10, help='number of tasks in continual learning')
parser.add_argument('--split', type=str, default='0.7-0.15-0.15', help='train/val/test split ratio')

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
DATA = args.dataset
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim

### set up logger
Path('./log/').mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
log_name = f'{get_ftime()}-{args.prefix}-{args.dataset}-{args.mask}'
fh = logging.FileHandler(f'./log/{log_name}.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

N, E, task_dataloader_l, train_ngh_finder_l, full_ngh_finder_l, nfeat, efeat = get_continual_data(args)

METRICS = ['AUC', 'AP', 'F1', 'RECALL']
mean_auc_l = []
mean_bf_l = []

def train_epoch(tgan, lr_model, train_loader, optimizer):
  tgan = tgan.train()             
  lr_model = lr_model.train()
  # batch loop
  for k in range(train_loader.batch_num):
    src_l_batch, dst_l_batch, ts_l_batch, label_l_batch = train_loader.load(k)

    optimizer.zero_grad()
    
    src_embed = tgan.tem_conv(src_l_batch, ts_l_batch, NODE_LAYER)

    src_label = torch.from_numpy(label_l_batch).float().to(device)
    lr_prob, h1, h2 = lr_model(src_embed)
    lr_prob = lr_prob.sigmoid()

    lr_criterion = torch.nn.BCELoss()
    lr_loss = lr_criterion(lr_prob, src_label)
    lr_loss.backward()
    
    optimizer.step()

def eval_epoch(tgan, lr_model, loader):
  tgan = tgan.eval()             
  lr_model = lr_model.eval()
  
  prob_l = []
  label_l = loader.label_l
  with torch.no_grad():
    for k in range(loader.batch_num):
      src_l_batch, dst_l_batch, ts_l_batch, _ = loader.load(k)
      
      src_embed = tgan.tem_conv(src_l_batch, ts_l_batch, NODE_LAYER)

      lr_prob, h1, h2 = lr_model(src_embed)
      lr_prob = lr_prob.sigmoid()
      prob_l.extend(lr_prob.cpu().numpy())
  
    fpr, tpr, thresholds = roc_curve(label_l, prob_l)
    # auc_score = auc(fpr, tpr)
    # 找到最佳阈值对应的索引
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    pred_labels = prob_l > optimal_threshold
    f1 = f1_score(label_l, pred_labels)
    recall = recall_score(label_l, pred_labels)
    auc_roc = roc_auc_score(label_l, prob_l)
    ap = average_precision_score(label_l, prob_l)
    metrics = (auc_roc, ap, f1, recall)
  
  return metrics

def evaluate_cont(auc_matrix):
  final_auc_l = np.array(auc_matrix[-1])
  # # TODO: 应该排除还是视为auc=1??
  # idx = np.where(np.array(final_auc_l) >= 0)[0]   # 算均值时排除没有负样本的task
  # final_auc_l = final_auc_l[idx]
  mean_auc = sum(final_auc_l)/len(final_auc_l)
  mean_auc = round(mean_auc, 4)
  
  backward_forget_l = []
  for i_task in range(len(auc_matrix[0])):
    bf = auc_matrix[i_task][i_task] - auc_matrix[-1][i_task]
    backward_forget_l.append(bf)
  mean_bf = round(np.mean(backward_forget_l), 4)
  
  return mean_auc, mean_bf

def get_task_perf_str(auc_matrix, i_task):
  auc_l = auc_matrix[i_task][:i_task+1]
  assert len(auc_l) == i_task + 1
  perf_str = ''
  for j_task in range(i_task + 1):  # 0, 1, ..., i_task
    perf_str += f'T{str(j_task).zfill(2)}:{auc_l[j_task]:.4f}|'
  mean_auc = sum(auc_l)/len(auc_l)
  perf_str += f' mean={mean_auc:.4f}'
  
  return perf_str

mean_auc_l, mean_bf_l = [], []
''' Run Loop '''
for i_run in range(args.n_run):
  logger.debug(f'===== ===== ===== Run#{i_run} Begin')
  device = torch.device('cuda:{}'.format(GPU))
  tgan = TGAN(train_ngh_finder_l[0], nfeat, efeat, num_layers=NUM_LAYER,
              attn_mode=ATTN_MODE, use_time=USE_TIME, agg_method=AGG_METHOD,
              seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT,
              node_dim=NODE_DIM, time_dim=TIME_DIM)   # node_dim & time_dim no use
  tgan = tgan.to(device)
  # decoder
  lr_model = LR(nfeat.shape[1])
  lr_model = lr_model.to(device)
  lr_criterion = torch.nn.BCELoss()
  
  auc_matrix = [[0.5 for _ in range(args.num_task)]
                for _ in range(args.num_task)]  # [T, T]
  ''' Task Loop '''
  for i_task, (train_loader, val_loader, test_loader) in enumerate(task_dataloader_l):
    train_ngh_finder = train_ngh_finder_l[i_task]
    full_ngh_finder = full_ngh_finder_l[i_task]
    
    lr_optimizer = torch.optim.Adam(lr_model.parameters(), lr=args.lr)

    best_epoch = [0, 0, 0, 0]
    best_val_metric = [0, 0, 0, 0]
    best_test_metric = [0, 0, 0, 0]
    max_test_metric = [0, 0, 0, 0]
    ''' Epoch Loop '''
    for epoch in tqdm(range(args.n_epoch)):
      logger.debug(f'===== ===== Epoch#{epoch} (Run#{i_run}, Task#{i_task})')
      tgan.ngh_finder = train_ngh_finder
      train_epoch(tgan, lr_model, train_loader, lr_optimizer)
      
      tgan.ngh_finder = full_ngh_finder
      val_metrics = eval_epoch(tgan, lr_model, val_loader)
      test_metrics = eval_epoch(tgan, lr_model, test_loader)
      
      val_auc, val_ap, val_f1, val_recall = val_metrics
      test_auc, test_ap, test_f1, test_recall = test_metrics
      
      logger.info(f'val_auc: {val_auc:.4f}, val_ap: {val_ap:.4f}, val_f1: {val_f1:.4f}, val_recall: {val_recall:.4f}')
      logger.info(f'test_auc: {test_auc:.4f}, test_ap: {test_ap:.4f}, test_f1: {test_f1:.4f}, test_recall: {test_recall:.4f}')
    ''' Epoch Loop End '''
    logger.debug(f'======== Run#{i_run} Task#{i_task} Training End ========')
    logger.debug(f'Evaluating previous tasks...')
    for j_task in range(i_task + 1):
      ''' 只验证test set '''
      metrics = eval_epoch(tgan, lr_model, task_dataloader_l[j_task][2])
      logger.debug(f'j_task={j_task}, auc={metrics[0]}')
      auc_matrix[i_task][j_task] = metrics[0]
    logger.debug(get_task_perf_str(auc_matrix, i_task))
  ''' Task Loop End '''
  logger.debug(f'========================== Run#{i_run} END ==========================')
  mean_auc, mean_bf = evaluate_cont(auc_matrix)
  logger.debug(f'Run#{i_run}: mean_auc={mean_auc:.4f}, mean_bf={mean_bf:.4f}')
  mean_auc_l.append(mean_auc)
  mean_bf_l.append(mean_bf)
''' Run Loop End '''

logger.debug(f'========================== SUMMARY ==========================')
for i_run in range(args.n_run):
  logger.debug(f'RUN#{i_run}: mean_auc={mean_auc_l[i_run]:.4f}, ' + 
               f'mean_bf={mean_bf_l[i_run]:.4f}')
  