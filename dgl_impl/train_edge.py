"""
RUN ME IN {project_foler}/dgl_impl/

python train_edge.py -d wikipedia
"""
import sys

import numpy as np
import torch
import dgl

import argparse
import time
import psutil

from dgl.data.utils import load_graphs
from pathlib import Path

from dataloader.samplers import MultiLayerTemporalNeighborSampler

parser = argparse.ArgumentParser('TGAT DGL train_edge TEST')
parser.add_argument('-d', '--dataset', type=str, choices=["wikipedia", "reddit"],
                    default='wikipedia')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
args = parser.parse_args()

GRAPH_PATH = f'./processed_data/{args.dataset}.bin'

g = load_graphs(GRAPH_PATH)[0][0]
print(g)

sampler = MultiLayerTemporalNeighborSampler(args, [20])
sampler.ts = g.ndata['timestamp'][10]
print(f'sampler.ts={sampler.ts}')

# sampler = dgl.dataloading.BlockSampler(1, False)

edges = g.in_edges(9, form='eid')
print(edges)

blocks = sampler.sample_blocks(g, [9, 10])
print(blocks)
