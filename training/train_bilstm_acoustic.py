import os
import json
import pickle
import random

import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.utils.class_weight import compute_class_weight

from models.bilstm import EmotionBiLSTM
from utils.frame_bilstm import (
    FrameDialogueDataset,
    frame_collate_fn,
    EmotionBiLSTMWithFrameEncoder
)


class FocalLoss(nn.Module):

    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):

        valid = targets != self.ignore_index

        logits = logits[valid]
        targets = targets[valid]

        log_probs = torch.log_softmax(logits, dim=1)

        target_log_probs = log_probs.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)

        target_probs = target_log_probs.exp()

        loss = -((1 - target_probs) ** self.gamma) * target_log_probs

        if self.alpha is not None:
            loss = loss * self.alpha[targets]

        return loss.mean()


EMBED_DIR = config.EMBEDDINGS_DIR

OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "bilstm_acoustic")
CHECKPOINT_DIR = os.path.join(config.CHECKPOINT_DIR, "bilstm_acoustic")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using:", device)


# Seed

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# Load per-utterance frame-level acoustic sequences (T_i, 96), already
# StandardScaler-normalized (fit on pooled train frames only) by
# utils/extract_acoustic_features.py. Deliberately NOT the flat mean/std
# aggregate -- MELD has utterances as short as ~4 frames, where mean/std
# is unstable; FrameEncoder (an LSTM) learns the temporal pooling instead.

train_frames = np.load(
    os.path.join(EMBED_DIR, "train_acoustic_frames.npy"), allow_pickle=True
)

dev_frames = np.load(
    os.path.join(EMBED_DIR, "dev_acoustic_frames.npy"), allow_pickle=True
)

frame_dim = train_frames[0].shape[1]

print("Train dialogues source utterances:", len(train_frames))
print("Dev dialogues source utterances:", len(dev_frames))
print("Frame feature dim:", frame_dim)


train_meta = pd.read_csv(os.path.join(EMBED_DIR, "train_metadata.csv"))
dev_meta = pd.read_csv(os.path.join(EMBED_DIR, "dev_metadata.csv"))


# Label encoder (shared, fit on train only by train_mlp.py)

encoder = joblib.load(os.path.join(EMBED_DIR, "label_encoder.pkl"))

print("Classes:", encoder.classes_)


train_dataset = FrameDialogueDataset(train_frames, train_meta, encoder)
dev_dataset = FrameDialogueDataset(dev_frames, dev_meta, encoder)


train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=frame_collate_fn
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=frame_collate_fn
)


# Model

model = EmotionBiLSTMWithFrameEncoder(
    bilstm_cls=EmotionBiLSTM,
    frame_input_dim=frame_dim,
    frame_hidden_dim=128,
    bilstm_hidden_dim=256,
    num_classes=len(encoder.classes_)
)

model.to(device)


# Class weights

labels = encoder.transform(train_meta["emotion"])

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

weights = torch.tensor(weights, dtype=torch.float32).to(device)


criterion = FocalLoss(alpha=weights, gamma=2.0, ignore_index=-100)

assert criterion.alpha is not None, "Class weights must reach FocalLoss"
print("Class weights reaching FocalLoss:", criterion.alpha)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,
    weight_decay=1e-3
)


epochs = 40

patience = 8

counter = 0

best_f1 = 0

best_path = os.path.join(CHECKPOINT_DIR, "bilstm_acoustic_best.pt")


# Training

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for padded_frames, frame_lengths, utt_lengths, y in train_loader:

        padded_frames = padded_frames.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(padded_frames, frame_lengths, utt_lengths)

        loss = criterion(
            output.reshape(-1, len(encoder.classes_)),
            y.reshape(-1)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()


    # Validation

    model.eval()

    predictions = []
    ground_truth = []

    with torch.no_grad():

        for padded_frames, frame_lengths, utt_lengths, y in dev_loader:

            padded_frames = padded_frames.to(device)

            output = model(padded_frames, frame_lengths, utt_lengths)

            pred = torch.argmax(output, dim=-1).cpu()

            mask = y != -100

            predictions.extend(pred[mask].numpy())
            ground_truth.extend(y[mask].numpy())


    accuracy = accuracy_score(ground_truth, predictions)
    macro_f1 = f1_score(ground_truth, predictions, average="macro")
    weighted_f1 = f1_score(ground_truth, predictions, average="weighted")

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {total_loss:.4f} | "
        f"Accuracy: {accuracy:.4f} | "
        f"Macro F1: {macro_f1:.4f} | "
        f"Weighted F1: {weighted_f1:.4f}"
    )

    if macro_f1 > best_f1:

        best_f1 = macro_f1
        counter = 0

        torch.save(model.state_dict(), best_path)

        print("Best model saved")

    else:

        counter += 1

        print(f"No improvement ({counter}/{patience})")

        if counter >= patience:

            print("\nEarly stopping triggered.\n")
            break

epochs_run = epoch + 1


# Load best model

model.load_state_dict(torch.load(best_path, map_location=device))
model.eval()

predictions = []
ground_truth = []

with torch.no_grad():

    for padded_frames, frame_lengths, utt_lengths, y in dev_loader:

        padded_frames = padded_frames.to(device)

        output = model(padded_frames, frame_lengths, utt_lengths)

        pred = torch.argmax(output, dim=-1).cpu()

        mask = y != -100

        predictions.extend(pred[mask].numpy())
        ground_truth.extend(y[mask].numpy())


print("\nBest Model Evaluation\n")

report = classification_report(
    ground_truth,
    predictions,
    target_names=encoder.classes_,
    zero_division=0
)

print(report)

matrix = confusion_matrix(ground_truth, predictions)


with open(
    os.path.join(OUTPUT_DIR, "classification_report.txt"),
    "w"
) as file:

    file.write(report)


np.save(
    os.path.join(OUTPUT_DIR, "confusion_matrix.npy"),
    matrix
)


prediction_df = pd.DataFrame(
    {
        "True Label": encoder.inverse_transform(ground_truth),
        "Predicted Label": encoder.inverse_transform(predictions)
    }
)

prediction_df.to_csv(
    os.path.join(OUTPUT_DIR, "predictions.csv"),
    index=False
)


results = {

    "model": "bilstm_acoustic",

    "embedding_strategy": "acoustic_frame_level_lstm_pooled",

    "accuracy": accuracy_score(ground_truth, predictions),
    "macro_f1": f1_score(ground_truth, predictions, average="macro"),
    "weighted_f1": f1_score(ground_truth, predictions, average="weighted"),

    "best_macro_f1": best_f1,

    "num_training_samples": len(train_dataset),
    "num_validation_samples": len(dev_dataset),

    "epochs_run": epochs_run,
    "patience": patience,

    "frame_input_dim": frame_dim,

    "classes": list(encoder.classes_)

}

with open(
    os.path.join(OUTPUT_DIR, "results.json"),
    "w"
) as file:

    json.dump(results, file, indent=4)


print("\nOutputs saved to:", OUTPUT_DIR)
print("✓ classification_report.txt")
print("✓ confusion_matrix.npy")
print("✓ predictions.csv")
print("✓ results.json")
