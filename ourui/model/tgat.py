import logging

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

class MergeLayer(torch.nn.Module):
    """
        Edge Decoder: 2-layer MLP
        Merge 2 nodes to obtain edge info
        dim1 + dim2 ==> dim3 ==> dim4
    """
    def __init__(self, dim1, dim2, dim3, dim4):
        """
            fc1:
                input_dim = dim1 + dim2
                out = dim3
            fc2:
                input = dim3
                output = dim4
        """
        super().__init__()
        # self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        # x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class TGAT(nn.Module):
    """ TGAT模型，DGL实现 """
    def __int__(self, feat_dim, emb_dim,
                num_layers=3, n_head=4, attn_drop=0.1):
        """
            TGAT模型初始化
            ==========================
            Params:
                feat_dim: 原始特征维度，假设节点特征维度=边特征维度
                emb_dim: 嵌入维度，假设隐层与最终嵌入维度相同
            ==========================
            Returns:
                None
        """
        self.feat_dim = feat_dim
        self.emb_dim = emb_dim
        self.time_encoder = None  # TODO:TE
        self.attn_list = torch.nn.ModuleList()
        for i in range(num_layers):
            input_dim = feat_dim if i == 0 else emb_dim
            self.attn_list.append(nn.MultiheadAttention(
                embed_dim=input_dim, kdim=emb_dim, vdim=emb_dim,
                num_heads=n_head, bias=False, dropout=attn_drop))


    def forward(self, blocks):
        """
            TGAT前向传播
            ==========================
            Params:
                blocks: 采样后的block，shape=[self.num_layers]
            ==========================
            Returns:
                emb: 节点嵌入
        """



