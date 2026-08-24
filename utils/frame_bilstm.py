"""
Shared plumbing for feeding frame-level acoustic sequences into the
unmodified EmotionBiLSTM.

The existing EmotionBiLSTM (models/bilstm.py) runs over the sequence of
utterances *within a dialogue* -- each utterance is already a single
feature vector. The acoustic-only tier instead has, per utterance, a
variable-length sequence of *frames* (down to ~4 frames for the shortest
MELD clips), and the task explicitly calls for letting an LSTM learn the
temporal pooling instead of flattening via mean/std.

FrameEncoder does that pooling (a small LSTM over an utterance's frames,
final hidden state as the pooled vector) so EmotionBiLSTM itself stays
completely unmodified -- it just receives a different input_dim, the same
pattern used by TrainableLayerWeighting for the wav2vec2 layer-weighted
tier.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset


class FrameEncoder(nn.Module):

    def __init__(self, input_dim=96, hidden_dim=128):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

    def forward(self, padded_frames, frame_lengths):

        packed = pack_padded_sequence(
            padded_frames,
            frame_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (h_n, _) = self.lstm(packed)

        return h_n[-1]


class FrameDialogueDataset(Dataset):
    """Groups per-utterance frame sequences into per-dialogue lists, same
    grouping/sort logic as DialogueDataset in training/train_bilstm.py."""

    def __init__(self, frame_array, metadata, encoder):

        self.dialogues = []

        metadata = metadata.copy()
        metadata["label"] = encoder.transform(metadata["emotion"])

        for _, dialogue in metadata.groupby("dialogue_id", sort=False):

            dialogue = dialogue.sort_values("utterance_id")
            indices = dialogue.index.tolist()

            frame_list = [
                torch.tensor(frame_array[i], dtype=torch.float32)
                for i in indices
            ]

            y = torch.tensor(dialogue["label"].values, dtype=torch.long)

            self.dialogues.append((frame_list, y))

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        return self.dialogues[idx]


def frame_collate_fn(batch):
    """Flattens frame sequences across all utterances in the batch (needed
    so FrameEncoder can pack/pad them in one LSTM pass), then reassembles
    the pooled per-utterance vectors back into per-dialogue sequences via
    utt_lengths in EmotionBiLSTMWithFrameEncoder.forward."""

    utt_lengths = torch.tensor([len(frame_list) for frame_list, _ in batch])

    all_frames = [
        frames
        for frame_list, _ in batch
        for frames in frame_list
    ]

    frame_lengths = torch.tensor([frames.shape[0] for frames in all_frames])
    padded_frames = pad_sequence(all_frames, batch_first=True)

    labels = pad_sequence(
        [y for _, y in batch],
        batch_first=True,
        padding_value=-100
    )

    return padded_frames, frame_lengths, utt_lengths, labels


class EmotionBiLSTMWithFrameEncoder(nn.Module):

    def __init__(
        self,
        bilstm_cls,
        frame_input_dim=96,
        frame_hidden_dim=128,
        bilstm_hidden_dim=256,
        num_classes=7
    ):
        super().__init__()

        self.frame_encoder = FrameEncoder(
            input_dim=frame_input_dim,
            hidden_dim=frame_hidden_dim
        )

        self.bilstm = bilstm_cls(
            input_dim=frame_hidden_dim,
            hidden_dim=bilstm_hidden_dim,
            num_classes=num_classes
        )

    def forward(self, padded_frames, frame_lengths, utt_lengths):

        pooled = self.frame_encoder(padded_frames, frame_lengths)

        dialogue_seqs = torch.split(pooled, utt_lengths.tolist())
        padded_utts = pad_sequence(dialogue_seqs, batch_first=True)

        return self.bilstm(padded_utts, utt_lengths)
