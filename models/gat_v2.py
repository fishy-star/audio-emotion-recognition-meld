import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GATv2Conv, PairNorm


def sinusoidal_positions(num_nodes, dim, device):

    position = torch.arange(num_nodes, device=device).unsqueeze(1).float()

    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device).float()
        * (-math.log(10000.0) / dim)
    )

    pe = torch.zeros(num_nodes, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: dim // 2 + dim % 2])

    return pe


class FlexibleEmotionGAT(nn.Module):

    def __init__(
        self,
        input_dim=768,
        hidden_dim=128,
        num_classes=7,
        heads=4,
        dropout=0.4,
        conv_type="gatv2",
        norm_type="layernorm",
        use_jk=False,
        use_pos_encoding=False,
        num_wav2vec_layers=None
    ):

        super().__init__()

        self.use_jk = use_jk
        self.use_pos_encoding = use_pos_encoding
        self.num_wav2vec_layers = num_wav2vec_layers

        if num_wav2vec_layers is not None:
            self.layer_weights = nn.Parameter(torch.zeros(num_wav2vec_layers))

        conv_cls = GATv2Conv if conv_type == "gatv2" else GATConv

        self.gat1 = conv_cls(
            input_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout
        )

        self.gat2 = conv_cls(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            concat=False,
            dropout=dropout
        )

        self.res1 = nn.Linear(input_dim, hidden_dim * heads)
        self.res2 = nn.Linear(hidden_dim * heads, hidden_dim)

        if norm_type == "pairnorm":
            self.norm1 = PairNorm()
            self.norm2 = PairNorm()
        else:
            self.norm1 = nn.LayerNorm(hidden_dim * heads)
            self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        if use_jk:
            self.jk_proj = nn.Linear(hidden_dim * heads + hidden_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index):

        if self.num_wav2vec_layers is not None:
            # x: (num_nodes, num_layers, dim) -> learned softmax-weighted fusion (Pepino et al. 2021)
            layer_softmax = torch.softmax(self.layer_weights, dim=0)
            x = (x * layer_softmax.view(1, -1, 1)).sum(dim=1)

        if self.use_pos_encoding:
            x = x + sinusoidal_positions(x.size(0), x.size(1), x.device)

        res1 = self.res1(x)
        h1 = self.gat1(x, edge_index)
        h1 = F.elu(h1)
        h1 = h1 + res1
        h1 = self.norm1(h1)
        h1 = self.dropout(h1)

        res2 = self.res2(h1)
        h2 = self.gat2(h1, edge_index)
        h2 = F.elu(h2)
        h2 = h2 + res2
        h2 = self.norm2(h2)
        h2 = self.dropout(h2)

        if self.use_jk:
            out = self.jk_proj(torch.cat([h1, h2], dim=1))
            out = F.elu(out)
        else:
            out = h2

        return self.classifier(out)


def drop_edges(edge_index, p=0.15, training=True):

    if not training or p <= 0:
        return edge_index

    self_loop_mask = edge_index[0] == edge_index[1]

    keep_mask = torch.rand(edge_index.size(1), device=edge_index.device) >= p

    keep_mask = keep_mask | self_loop_mask

    return edge_index[:, keep_mask]
