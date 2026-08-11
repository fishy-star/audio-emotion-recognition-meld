import torch
import torch.nn as nn


class EmotionBiLSTM(nn.Module):

    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        num_layers=2,
        num_classes=7
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )


        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(

    nn.Linear(
        hidden_dim * 2,
        256
    ),

    nn.ReLU(),

    nn.Dropout(0.4),

    nn.Linear(
        256,
        num_classes
    )
)


    def forward(
        self,
        x,
        lengths
    ):

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )


        packed_out, _ = self.lstm(
            packed
        )


        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True
        )


        lstm_out = self.dropout(
            lstm_out
        )


        output = self.classifier(
            lstm_out
        )


        return output