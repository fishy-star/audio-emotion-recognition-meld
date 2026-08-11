import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv


class ProposedTypedEmotionGAT(nn.Module):

    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        num_classes=7,
        num_relations=4,
        heads=4,
        dropout=0.2
    ):

        super().__init__()

        self.num_relations = num_relations

        # Speaker embedding
        self.speaker_embedding = nn.Embedding(
            num_embeddings=20,
            embedding_dim=32
        )

        self.input_proj = nn.Linear(
            input_dim + 32,
            hidden_dim
        )

        self.residual = nn.Linear(
            input_dim + 32,
            hidden_dim
        )

        self.conv1 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=num_relations
        )

        self.conv2 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=num_relations
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

            nn.Linear(
                hidden_dim,
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
        edge_index,
        edge_type,
        speaker_ids
    ):

        edge_attr = F.one_hot(
            edge_type,
            num_classes=self.num_relations
        ).float()

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

        x_proj = self.input_proj(
            x
        )

        residual = self.residual(
            x
        )

        x = self.conv1(
            x_proj,
            edge_index,
            edge_attr=edge_attr
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
            edge_attr=edge_attr
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