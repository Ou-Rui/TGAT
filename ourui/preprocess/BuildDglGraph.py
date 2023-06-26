"""
copy from APAN
RUN ME IN {project_foler}/ourui/

python preprocess/BuildDglGraph.py -d wikipedia
"""


import dgl
import torch
import argparse
import pandas as pd
import numpy as np
from dgl.data.utils import save_graphs

import dgl.function as fn

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('-d', '--dataset', type=str, choices=["wikipedia", "reddit"],
                    help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--new_node_count', action='store_true',
                    help='count how many nodes are not in training set')
args = parser.parse_args()
args.new_node_count = True

GRAPH_DF_PATH = f'./processed_data/{args.dataset}.csv'
EDGE_FEAT_PATH = f'./processed_data/{args.dataset}.npy'
OUT_PATH = f'./processed_data/{args.dataset}.bin'

graph_df = pd.read_csv(GRAPH_DF_PATH)
edge_features = np.load(EDGE_FEAT_PATH)
nfeat_dim = edge_features.shape[1]

src = torch.tensor(graph_df.u.values)
dst = torch.tensor(graph_df.i.values)
label = torch.tensor(graph_df.label.values, dtype=torch.float32)
timestamp = torch.tensor(graph_df.ts.values, dtype=torch.float32)
edge_feat = torch.tensor(edge_features[1:], dtype=torch.float32)

g = dgl.graph((torch.cat([src, dst]), torch.cat([dst, src])))
len_event = src.shape[0]

g.edata['label'] = label.repeat(2).squeeze()
g.edata['timestamp'] = timestamp.repeat(2).squeeze()
g.edata['feat'] = edge_feat.repeat(2, 1).squeeze()

print(g)
save_graphs(OUT_PATH, g)

if args.new_node_count:
    origin_num_edges = g.num_edges() // 2
    train_eid = torch.arange(0, int(0.7 * origin_num_edges))
    un_train_eid = torch.arange(int(0.7 * origin_num_edges), origin_num_edges)

    train_g = dgl.graph(g.find_edges(train_eid))
    val_n_test_g = dgl.compact_graphs(dgl.graph(g.find_edges(un_train_eid)))

    print(
        f'total nodes: {g.num_nodes()}, training nodes: {train_g.num_nodes()}, val_n_test nodes: {val_n_test_g.num_nodes()}')
    old_nodes = val_n_test_g.num_nodes() - g.num_nodes() + train_g.num_nodes()
    print(f'old nodes in val_n_test: {old_nodes} ({round(old_nodes * 100 / val_n_test_g.num_nodes(), 4)}%)')
    new_nodes = g.num_nodes() - train_g.num_nodes()
    print(f'new nodes in val_n_test: {new_nodes} ({round(new_nodes * 100 / val_n_test_g.num_nodes(), 4)}%)')
