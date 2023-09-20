"""
copy from APAN
RUN ME IN {project_foler}/dgl_impl/

python preprocess/preprocess_csv.py -d wikipedia
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from torch._C import dtype
from tqdm import tqdm
import csv
import os


parser = argparse.ArgumentParser('Interface for data preprocessing')
parser.add_argument('-d', '--dataset', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
args = parser.parse_args()

DATASET_PATH = "/root/ourui/Datasets"
OUT_DF = f'./processed_data/{args.dataset}.csv'
OUT_FEAT = f'./processed_data/{args.dataset}.npy'


def preprocess(args):
    # 如果processed_data目录不存在，则创建；存在就忽略
    Path("processed_data/").mkdir(parents=True, exist_ok=True)

    if args.dataset in ['wikipedia', 'reddit']:
        dataset_path = f'{DATASET_PATH}/{args.dataset}.csv'
        feat_dim = 172
    else:
        raise ValueError('Please check the dataset name.')

    u_list, i_list, ts_list, label_list, idx_list = [], [], [], [], []
    feat_l = []
    reindex = Reindex(args)
    with open(dataset_path) as f:
        header = next(f)
        print(header)
        for idx, line in enumerate(tqdm(f)):
            e = line.strip().split(',')

            u, i = int(e[0]), int(e[1])  # u=user=src, i=item=dst

            u, i = reindex.user2id(args, u, i)
            # dataset are sorted by timestamp
            ts = float(e[2])        # ts start from 0.0 (wikipedia)
            label = float(e[3])  # int(e[3])

            feat = [float(x) for x in e[4:]]

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)

    feat_l = np.array(feat_l, dtype='float32')

    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), feat_l


class Reindex(object):
    def __init__(self, args):
        super(Reindex, self).__init__()
        self.user_idx = {}
        self.item_idx = {}
        self.curr_idx = 0

    def bipartite_graph_reindex(self, u, i):
        """ wikipedia & reddit """
        if u not in self.user_idx.keys():
            self.user_idx[u] = self.curr_idx
            u = self.curr_idx
            self.curr_idx += 1
        else:
            u = self.user_idx[u]

        if i not in self.item_idx.keys():
            self.item_idx[i] = self.curr_idx
            i = self.curr_idx
            self.curr_idx += 1
        else:
            i = self.item_idx[i]

        return u, i

    def graph_reindex(self, u, i):
        """ used for alipay... """
        if u not in self.user_idx.keys():
            self.user_idx[u] = self.curr_idx
            u = self.curr_idx
            self.curr_idx += 1
        else:
            u = self.user_idx[u]
        if i not in self.user_idx.keys():
            self.user_idx[i] = self.curr_idx
            i = self.curr_idx
            self.curr_idx += 1
        else:
            i = self.user_idx[i]
        return u, i

    def user2id(self, args, u, i):
        """
            u=user: src_nid
            i=item: dst_nid
            ----
            return
                re-indexed nid
        """
        if args.dataset == 'alipay':
            u, i = self.graph_reindex(u, i)
        else:
            u, i = self.bipartite_graph_reindex(u, i)
        return u, i


graph, edge_feat = preprocess(args)

# 在feat顶部插入空白行？
empty = np.zeros(edge_feat.shape[1], dtype='float32')[np.newaxis, :]
feat = np.vstack([empty, edge_feat])

graph.to_csv(OUT_DF)
np.save(OUT_FEAT, feat)
