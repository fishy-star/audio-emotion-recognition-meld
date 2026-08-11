import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv


class ProposedEmotionGAT(nn.Module):

    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        num_classes=7,
        heads=4,
        dropout=0.3
    ):

        super().__init__()


        self.input_layer = nn.Linear(
            input_dim,
            hidden_dim
        )


        self.gat1 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout
        )


        self.norm1 = nn.LayerNorm(
            hidden_dim * heads
        )


        self.gat2 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            concat=False,
            dropout=dropout
        )


        self.norm2 = nn.LayerNorm(
            hidden_dim
        )


        self.residual = nn.Linear(
            input_dim,
            hidden_dim
        )


        self.dropout = nn.Dropout(
            dropout
        )


        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.LayerNorm(
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                num_classes
            )
        )


    def forward(
        self,
        x,
        edge_index
    ):

        residual = self.residual(
            x
        )


        x = self.input_layer(
            x
        )


        x = self.gat1(
            x,
            edge_index
        )


        x = self.norm1(
            x
        )


        x = F.elu(
            x
        )


        x = self.dropout(
            x
        )


        x = self.gat2(
            x,
            edge_index
        )


        x = x + residual


        x = self.norm2(
            x
        )


        x = F.elu(
            x
        )


        x = self.dropout(
            x
        )


        return self.classifier(
            x
        )