"""
RUN ME in {project_folder}/
python -u txn_process.py -d credit_full
"""
import json
import sys
import numpy as np
import pandas as pd
import argparse

from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler


parser = argparse.ArgumentParser('Data Statistic Analysis')
parser.add_argument('-d', '--dataset', type=str, help='dataset name', choices=['txn', 'txn_filter', 'credit_full'],
                    default='txn_filter')
args = parser.parse_args()




Path("datasets/").mkdir(parents=True, exist_ok=True)

if args.dataset in ['txn', 'txn_filter']:
  INPUT_DIC = '/remote-home/ourui/datasets/Transactions/'
  input_filenames = [f'output-d{i}.csv' for i in range(1, 7)]
else:
  INPUT_DIC = '/remote-home/ourui/datasets/credit_full/'
  input_filenames = [f'd{i}.csv' for i in range(1, 7)]
  
OUTPUT_DIC = './processed/'
output_filename = args.dataset
OUT_DF = OUTPUT_DIC + 'ml_{}.csv'.format(output_filename)
OUT_FEAT = OUTPUT_DIC + 'ml_{}.npy'.format(output_filename)
OUT_NODE_FEAT = OUTPUT_DIC + 'ml_{}_node.npy'.format(output_filename)

OUT_LOG_DIC = './draw_analyze/analysis_logs/'
Path(OUT_LOG_DIC).mkdir(parents=True, exist_ok=True)
OUT_LOG_PATH = OUT_LOG_DIC + f'{args.dataset}.txt'

def read_csv(log):
  uid_l, acc_limit_l, single_limit_l, ts_l, amount_l = [], [], [], [], []
  card_area_l, merchant_code_l, pre_trade_result_l, is_common_ip_l, phone_equal_l, sign_l = [], [], [], [], [], []
  
  fraud_user_set = set()

  for filename in input_filenames:
    len_org = len(uid_l)
    input_path = INPUT_DIC + filename
    with open(input_path) as f:
      s = next(f)   # 跳过第一行header
      for idx, line in enumerate(f):
        item = line.strip().split(',')
        
        uid = int(item[0])
        # eid = int(item[1])
        acc_limit = float(item[2])
        single_limit = float(item[3])
        ts = int(item[4])
        amount = float(item[5])
        card_area = float(item[6])
        merchant_code = float(item[7])      # merchant id
        # receiving_customer_code = float(item[8])    # same as merchant id ??
        pre_trade_result = int(float(item[9]))
        is_common_ip = int(item[10])
        phone_equal = int(item[11])
        sign = int(item[12])
        
        if args.dataset == 'txn_filter':
        # 忽略欺诈者在执行欺诈以后的交易记录
          if uid in fraud_user_set:
            continue
          if sign == 1:
            fraud_user_set.add(uid)
        
        uid_l.append(uid)
        acc_limit_l.append(acc_limit)
        single_limit_l.append(single_limit)
        ts_l.append(ts)
        amount_l.append(amount)
        card_area_l.append(card_area)
        merchant_code_l.append(merchant_code)
        pre_trade_result_l.append(pre_trade_result)
        is_common_ip_l.append(is_common_ip)
        phone_equal_l.append(phone_equal)
        sign_l.append(sign)
    log.write(f'file:{filename}, record_num={len(uid_l)-len_org} \n')
    
    
  return uid_l, acc_limit_l, single_limit_l, ts_l, amount_l, \
    card_area_l, merchant_code_l, pre_trade_result_l, \
    is_common_ip_l, phone_equal_l, sign_l
      
def analyze_basics(log, item_list, msg):
  min_val = np.min(item_list)
  max_val = np.max(item_list)
  mean_val = np.mean(item_list)
  std_val = np.std(item_list)
  # mean_val, std_val = 0, 0
  log.write(f'{msg}, min={min_val}, max={max_val}, mean={mean_val}, std={std_val} \n')

def nid_reidx(uid_l, merchant_code_l, log):
  """
  对用户、商家重新编号
  ==========================================
  Args:
    uid_l (List[int]): len=N
    merchant_code_l (List[float]): len=M
  Returns:
    new_uid_l (List[int]): len=N
    new_mid_l (List[int]): len=M
  """
  count = 1   # id从1开始, 而不是0
  new_uid_l = []
  uid_map = dict()    # map[int]=int
  for uid in uid_l:
    if uid in uid_map:
      new_uid = uid_map[uid]  
    else:
      uid_map[uid] = count
      new_uid = count
      count += 1
    new_uid_l.append(new_uid)
  
  new_mid_l = []
  mid_map = dict()    # map[float]=int
  for mid in merchant_code_l:
    if mid in mid_map:
      new_mid = mid_map[mid]  
    else:
      mid_map[mid] = count
      new_mid = count
      count += 1
    new_mid_l.append(new_mid)
  
  log.write(f"uid_reidx: unique_uid_num={len(uid_map)} \n")
  log.write(f"mid_reidx: unique_mid_num={len(mid_map)} \n")
  assert(np.min(new_uid_l) == 1)
  assert(np.max(new_uid_l) == len(uid_map))
  assert(np.max(new_uid_l) + 1 == np.min(new_mid_l))
  assert(np.max(new_mid_l) == len(uid_map) + len(mid_map))
  
  return new_uid_l, new_mid_l

def feat_extract(acc_limit_l, single_limit_l, ts_l, amount_l,
                 card_area_l, is_common_ip_l, phone_equal_l):
  """
  提取每条交易的特征向量
  =======================================
  Args:
    acc_limit_l (List[float]): 0.0~5000000.0, 跨度大
    single_limit_l (List[float]): 0.0~1000000.0, 跨度大
    ts_l (List[int]): 
    amount_l (List[float]): 0.0~19125000.0, 跨度大
    card_area_l (List[float]): 意义不明的浮点数
    is_common_ip_l (List[int]): 0/1取值
    phone_equal_l (List[int]): 0/1取值
  Returns:
    feat_l (List[ndarray]): [E][7]
  """
  # acc_limit_l: 对数 + Z-score标准化
  acc_limit_l = [x + 1 for x in acc_limit_l]
  acc_limit_l = np.log(acc_limit_l)
  acc_limit_l = StandardScaler().fit_transform(np.reshape(acc_limit_l, [-1, 1]))
  acc_limit_l = np.reshape(acc_limit_l, [-1])
  # single_limit_l: 对数 + Z-score标准化
  single_limit_l = [x + 1 for x in single_limit_l]
  single_limit_l = np.log(single_limit_l)
  single_limit_l = StandardScaler().fit_transform(np.reshape(single_limit_l, [-1, 1]))
  single_limit_l = np.reshape(single_limit_l, [-1])
  # ts_l: MinMax归一化
  ts_l = MinMaxScaler().fit_transform(np.reshape(ts_l, [-1, 1]))
  ts_l = np.reshape(ts_l, [-1])
  # amount_l: 对数 + Z-score标准化
  amount_l = [x + 1 for x in amount_l]
  amount_l = np.log(amount_l)
  amount_l = StandardScaler().fit_transform(np.reshape(amount_l, [-1, 1]))
  amount_l = np.reshape(amount_l, [-1])
  # is_common_ip_l, phone_equal_l: 暂时不变
  # is_common_ip_l = np.array(is_common_ip_l, dtype=float)
  # phone_equal_l = np.array(phone_equal_l, dtype=float)
  
  ''' 组装edge_feat '''
  feat_l = []
  for item in zip(acc_limit_l, single_limit_l, ts_l, amount_l,
                  card_area_l, is_common_ip_l, phone_equal_l):
    acc_limit, single_limit, ts, amount, card_area, is_common_ip, phone_equal = item
    feat = np.array([acc_limit, single_limit, ts, amount, card_area, is_common_ip, phone_equal])
    feat_l.append(feat)
    
  return feat_l
  
def main(log):
  uid_l, acc_limit_l, single_limit_l, ts_l, amount_l, \
      card_area_l, merchant_code_l, pre_trade_result_l, \
      is_common_ip_l, phone_equal_l, sign_l = read_csv(log)
  
  
  log.write(f'total record num = {len(uid_l)}\n')
  analyze_basics(log, uid_l, 'uid')
  analyze_basics(log, acc_limit_l, 'acc_limit')
  analyze_basics(log, single_limit_l, 'single_limit')
  analyze_basics(log, ts_l, 'ts')
  analyze_basics(log, amount_l, 'amount')
  analyze_basics(log, card_area_l, 'card_area')
  analyze_basics(log, merchant_code_l, 'merchant_code')
  analyze_basics(log, pre_trade_result_l, 'pre_trade_result')
  analyze_basics(log, is_common_ip_l, 'is_common_ip')
  analyze_basics(log, phone_equal_l, 'phone_equal')
  analyze_basics(log, sign_l, 'sign')
  
  uid_l, mid_l =  nid_reidx(uid_l, merchant_code_l, log)
  feat_l = feat_extract(acc_limit_l, single_limit_l, ts_l, amount_l,
                        card_area_l, is_common_ip_l, phone_equal_l)
  
  eid_l = range(1, len(uid_l) + 1)  # eid从1开始
  
  frame = pd.DataFrame({'u': uid_l,
                        'i': mid_l,
                        'ts': ts_l,
                        'label': sign_l,
                        'idx': eid_l})
  frame.to_csv(OUT_DF)
  
  # 开头填充空行, 因为eid从1开始
  feat_l = np.array(feat_l)
  empty = np.zeros(feat_l.shape[1])[np.newaxis, :]
  feat_l = np.vstack([empty, feat_l]) 
  np.save(OUT_FEAT, feat_l)
  
  max_idx = max(frame.u.max(), frame.i.max())
  node_feat_l = np.random.normal(size=(max_idx + 1, 128))   # 正态分布
  np.save(OUT_NODE_FEAT, node_feat_l)

with open(OUT_LOG_PATH, 'w') as log:
  main(log)

  
    
    
    