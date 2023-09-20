import dgl
import torch
import numpy as np
import random


def temporal_edges_filter(edges):
    pass

class MultiLayerTemporalNeighborSampler(dgl.dataloading.BlockSampler):
    """
        多层时域邻居采样器：考虑时间约束
    """
    def __init__(self, args, fanouts, return_eids=False):
        super().__init__(len(fanouts), return_eids)

        self.fanouts = fanouts  # List[n_layers]，表示每一层的采样数量
        self.ts = 0
        self.args = args
        self.frontiers = [None for _ in range(len(fanouts))]  # 每一层的in-neighbor



    def sample_frontier(self, block_id, g, seed_nodes):
        """
            会被父类BlockSampler中的sample_blocks()调用
            重写该函数来实现自定义采样
        """
        fanout = self.fanouts[block_id]

        g = dgl.in_subgraph(g, seed_nodes)
        ''' 仅保留edge_ts <= self.ts的边 '''
        # remove edges whose timestamp > ts of current batch
        g.remove_edges(torch.where(g.edata['timestamp'] > self.ts)[0])

        if fanout is None:
            frontier = g
            # frontier = dgl.in_subgraph(g, seed_nodes)
        else:
            if self.args.uniform:
                frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout)
            else:
                # sample neighbors based on timestamp
                frontier = dgl.sampling.select_topk(g, fanout, 'timestamp', seed_nodes)

        self.frontiers[block_id] = frontier
        return frontier
