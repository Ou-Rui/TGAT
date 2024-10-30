import os, sys
import math, random

import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from graph import NeighborFinder
from dataloader.loader import DataLoader


def get_taski_data(args, full_data, full_adj_list_prev, i_task, 
                   num_sample_per_task, split_list):
  """
  获取单个task的数据, called by get_continual_data
  ==========================
  Returns:
    loaders: [train, val, test]
    train_adj_list
    full_adj_list: 作为下一个task的prev
  """
  src_l, dst_l, e_idx_l, label_l, ts_l = full_data
  
  task_start_idx = i_task*num_sample_per_task
  task_end_idx = min((i_task+1)*num_sample_per_task, len(src_l))
  task_num_sample = task_end_idx - task_start_idx
  
  task_src_l = src_l[task_start_idx:task_end_idx]
  task_dst_l = dst_l[task_start_idx:task_end_idx]
  task_e_idx_l = e_idx_l[task_start_idx:task_end_idx]
  task_label_l = label_l[task_start_idx:task_end_idx]
  task_ts_l = ts_l[task_start_idx:task_end_idx]

  train_num_sample = int(split_list[0] * task_num_sample)
  val_num_sample = int(split_list[1] * task_num_sample)

  train_src_l = task_src_l[:train_num_sample]
  train_dst_l = task_dst_l[:train_num_sample]
  train_e_idx_l = task_e_idx_l[:train_num_sample]
  train_label_l = task_label_l[:train_num_sample]
  train_ts_l = task_ts_l[:train_num_sample]

  val_src_l = task_src_l[train_num_sample:train_num_sample+val_num_sample]
  val_dst_l = task_dst_l[train_num_sample:train_num_sample+val_num_sample]
  val_e_idx_l = task_e_idx_l[train_num_sample:train_num_sample+val_num_sample]
  val_label_l = task_label_l[train_num_sample:train_num_sample+val_num_sample]
  val_ts_l = task_ts_l[train_num_sample:train_num_sample+val_num_sample]

  test_src_l = task_src_l[train_num_sample+val_num_sample:]
  test_dst_l = task_dst_l[train_num_sample+val_num_sample:]
  test_e_idx_l = task_e_idx_l[train_num_sample+val_num_sample:]
  test_label_l = task_label_l[train_num_sample+val_num_sample:]
  test_ts_l = task_ts_l[train_num_sample+val_num_sample:]

  ''' 初始化adj_list '''
  max_idx = max(src_l.max(), dst_l.max())
  train_adj_list, full_adj_list = [[] for _ in range(max_idx + 1)], [[] for _ in range(max_idx + 1)]
  ''' 不包含过去的边 '''
  # if full_adj_list_prev is None:
  #   # 将过去task的边添加进来
  #   prev_src_l = src_l[:task_start_idx]
  #   prev_dst_l = dst_l[:task_start_idx]
  #   prev_e_idx_l = e_idx_l[:task_start_idx]
  #   prev_ts_l = ts_l[:task_start_idx]
  #   for src, dst, eidx, ts in zip(prev_src_l, prev_dst_l, prev_e_idx_l, prev_ts_l):
  #     train_adj_list[src].append((dst, eidx, ts))
  #     train_adj_list[dst].append((src, eidx, ts))
  #     full_adj_list[src].append((dst, eidx, ts))
  #     full_adj_list[dst].append((src, eidx, ts))
  # else:
  #   train_adj_list = full_adj_list_prev
  #   full_adj_list = full_adj_list_prev
    
  ''' 向adj_list中添加当前任务的边 '''
  for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    train_adj_list[src].append((dst, eidx, ts))
    train_adj_list[dst].append((src, eidx, ts))
  
  for src, dst, eidx, ts in zip(task_src_l, task_dst_l, task_e_idx_l, task_ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))

  train_loader = DataLoader(args, train_src_l, train_dst_l, train_ts_l, 
                            train_label_l, train_e_idx_l, mode='train')
  val_loader = DataLoader(args, val_src_l, val_dst_l, val_ts_l, 
                          val_label_l, val_e_idx_l, mode='val')
  test_loader = DataLoader(args, test_src_l, test_dst_l, test_ts_l, 
                           test_label_l, test_e_idx_l, mode='test')
  
  return [train_loader, val_loader, test_loader], train_adj_list, full_adj_list
  
def get_continual_data(args):
  """
  Args:
    args: dict
  Returns:
    N:
    E:
    task_loader_l: list[n_task]
    train_ngh_finder_l: list[n_task]
    full_ngh_finder_l: list[n_task]
    nfeat: [N, d]
    efeat: [E, d]

  """
  DATA_ROOT = './processed/'
  Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
  
  CONT_DATA_ROOT = DATA_ROOT + 'continual_data/'
  Path(CONT_DATA_ROOT).mkdir(parents=True, exist_ok=True)
  Path(CONT_DATA_ROOT+f'{args.dataset}/').mkdir(parents=True, exist_ok=True)

  split_str = args.split.split('-')
  split_list = list(map(float, split_str))
  assert len(split_list) == 3 and sum(split_list) == 1
  DATA_DIC = CONT_DATA_ROOT + f'{args.dataset}/' + \
      f'tr{split_list[0]}_val{split_list[1]}_test{split_list[2]}/'
  Path(DATA_DIC).mkdir(parents=True, exist_ok=True)

  g_df = pd.read_csv(f'./processed/ml_{args.dataset}.csv')
  efeat = np.load(f'./processed/ml_{args.dataset}.npy')
  nfeat = np.load(f'./processed/ml_{args.dataset}_node.npy')   # [N, 172]
  
  src_l = g_df.u.values
  dst_l = g_df.i.values
  e_idx_l = g_df.idx.values
  label_l = g_df.label.values
  ts_l = g_df.ts.values

  max_idx = max(src_l.max(), dst_l.max())
  N = max_idx + 1
  E = len(src_l)
  
  ''' 目前假设任务按照样本数量均分 '''
  num_sample_per_task = math.ceil(float(E)/args.num_task)
  task_dataloader_l = []    # len()=args.num_task, 每项包含train_loader, val, test
  train_ngh_finder_l = []
  full_ngh_finder_l = [] 
  full_adj_list = None
  
  for i_task in range(args.num_task):
    filename = DATA_DIC + f'{args.dataset}-{args.num_task}-{i_task}.pkl'
    if os.path.isfile(filename):
      with open(filename, 'rb') as f:
        task_dataloader, train_ngh_finder, full_ngh_finder = pickle.load(f)
      full_adj_list = None
    else:
      task_dataloader, train_adj_list, full_adj_list = get_taski_data(
          args, [src_l, dst_l, e_idx_l, label_l, ts_l],
          full_adj_list, i_task, num_sample_per_task, split_list)
      train_ngh_finder = NeighborFinder(train_adj_list, uniform=args.uniform)
      full_ngh_finder = NeighborFinder(full_adj_list, uniform=args.uniform)
      with open(filename, 'wb') as f:
        pickle.dump((task_dataloader, train_ngh_finder, full_ngh_finder), f)
    task_dataloader_l.append(task_dataloader)
    train_ngh_finder_l.append(train_ngh_finder)
    full_ngh_finder_l.append(full_ngh_finder)
  
  return N, E, task_dataloader_l, train_ngh_finder_l, full_ngh_finder_l, \
         nfeat, efeat


