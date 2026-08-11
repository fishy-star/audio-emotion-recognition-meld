import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import RGATConv


class ProposedRGATEmotion(nn.Module):

    def __init__(
        self,
        input_dim=784,
        hidden_dim=128,
        num_classes=7,
        num_relations=4,
        heads=4,
        dropout=0.2
    ):

        super().__init__()


        self.speaker_embedding = nn.Embedding(
            260,
            16
        )


        self.conv1 = RGATConv(
            input_dim,
            hidden_dim,
            num_relations=num_relations,
            heads=heads,
            concat=True,
            dropout=dropout
        )


        self.conv2 = RGATConv(
            hidden_dim * heads,
            hidden_dim,
            num_relations=num_relations,
            heads=1,
            concat=False,
            dropout=dropout
        )


        self.norm1 = nn.LayerNorm(
            hidden_dim * heads
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

            nn.ELU(),

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
        speaker_ids,
        edge_index,
        edge_type
    ):


        speaker_vec = self.speaker_embedding(
            speaker_ids
        )


        x = torch.cat(
            [
                x,
                speaker_vec
            ],
            dim=1
        )


        res = self.residual(
            x
        )


        x = self.conv1(
            x,
            edge_index,
            edge_type
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


        x = self.conv2(
            x,
            edge_index,
            edge_type
        )


        x = x + res


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