import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv


class EmotionGAT(nn.Module):

    def __init__(
        self,
        input_dim=768,
        hidden_dim=128,
        num_classes=7,
        heads=4,
        dropout=0.4
    ):

        super().__init__()


        self.gat1 = GATConv(
            input_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout
        )


        self.gat2 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            concat=False,
            dropout=dropout
        )


        self.res1 = nn.Linear(
            input_dim,
            hidden_dim * heads
        )


        self.res2 = nn.Linear(
            hidden_dim * heads,
            hidden_dim
        )


        self.norm1 = nn.LayerNorm(
            hidden_dim * heads
        )


        self.norm2 = nn.LayerNorm(
            hidden_dim
        )


        self.dropout = nn.Dropout(
            dropout
        )


        self.classifier = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, num_classes)
)



    def forward(
        self,
        x,
        edge_index
    ):

        res1 = self.res1(
            x
        )


        h = self.gat1(
            x,
            edge_index
        )


        h = F.elu(
            h
        )


        h = h + res1


        h = self.norm1(
            h
        )


        h = self.dropout(
            h
        )


        res2 = self.res2(
            h
        )


        h2 = self.gat2(
            h,
            edge_index
        )


        h2 = F.elu(
            h2
        )


        h2 = h2 + res2


        h2 = self.norm2(
            h2
        )


        h2 = self.dropout(
            h2
        )


        return self.classifier(
            h2
        )
