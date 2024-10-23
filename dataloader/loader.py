import math
import random

import numpy as np


class DataLoader:
  def __init__(self, args, src_l, dst_l, ts_l, label_l, e_idx_l,
               mode='train'):
    self.src_l = src_l
    self.dst_l = dst_l
    self.ts_l = ts_l
    self.label_l = label_l
    self.e_idx_l = e_idx_l

    self.bs = args.bs
    self.E = len(src_l)
    self.batch_num = math.ceil(self.E / self.bs)
    
    self.mode = mode
  
  def load(self, batch_idx):
    """
    Args:
      batch_idx
    Returns:
      src_l_batch: [b]
      dst_l_batch: [b]
      ts_l_batch: [b]
      label_l_batch: [b]
    """
    s_idx = batch_idx * self.bs  # 当前batch的第一个节点在label_index中的idx
    e_idx = min(self.E, s_idx + self.bs)  # 最后一个节点在label_index中的idx
    b = e_idx - s_idx
    
    src_l_batch = self.src_l[s_idx:e_idx]
    dst_l_batch = self.dst_l[s_idx:e_idx]
    ts_l_batch = self.ts_l[s_idx:e_idx]
    label_l_batch = self.label_l[s_idx:e_idx]
    
    return src_l_batch, dst_l_batch, ts_l_batch, label_l_batch
  