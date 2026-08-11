import torch
import torch.nn as nn


class EmotionMLP(nn.Module):

    def __init__(self, input_dim=768, num_classes=7):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )


    def forward(self, x):
        return self.network(x)