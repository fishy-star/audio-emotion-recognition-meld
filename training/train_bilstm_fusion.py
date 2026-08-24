"""
BiLSTM fusion tier: concatenates a pooled acoustic summary onto each
wav2vec2 utterance vector, rather than fusing at the raw acoustic frame
level. The dialogue-level EmotionBiLSTM already consumes one vector per
utterance (not per frame), so the acoustic side needs to be pooled to a
single vector per utterance regardless of fusion point -- fusing the
already-computed flat aggregate (mean+std, 192-dim) onto the layer-weighted
wav2vec2 vector (768-dim) reuses the existing DialogueDataset/collate_fn
machinery unchanged and needs no new frame-level plumbing, whereas fusing
at the raw frame level would require redoing the hierarchical FrameEncoder
wiring from train_bilstm_acoustic.py just to make room for a second input
stream. Chosen for less architecture change, per the task's own guidance.
"""

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

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from models.bilstm import EmotionBiLSTM
from utils.embedding_strategies import TrainableLayerWeighting


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


class EmotionBiLSTMWithLayerWeightingFusion(nn.Module):

    def __init__(self, wav2vec_dim=768, acoustic_dim=192, hidden_dim=256, num_classes=7, num_layers=13):
        super().__init__()

        self.layer_weighting = TrainableLayerWeighting(num_layers=num_layers)
        self.bilstm = EmotionBiLSTM(
            input_dim=wav2vec_dim + acoustic_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )

    def forward(self, x_wav2vec, x_acoustic, lengths):
        # x_wav2vec: (batch, seq_len, num_layers, hidden_dim)
        # x_acoustic: (batch, seq_len, acoustic_dim)
        pooled = self.layer_weighting(x_wav2vec)
        fused = torch.cat([pooled, x_acoustic], dim=-1)
        return self.bilstm(fused, lengths)


class DialogueDataset(Dataset):

    def __init__(self, wav2vec_embeddings, acoustic_embeddings, metadata, encoder):

        self.dialogues = []

        metadata = metadata.copy()
        metadata["label"] = encoder.transform(metadata["emotion"])

        for _, dialogue in metadata.groupby("dialogue_id", sort=False):

            dialogue = dialogue.sort_values("utterance_id")
            indices = dialogue.index.tolist()

            x_wav = wav2vec_embeddings[indices]
            x_ac = acoustic_embeddings[indices]

            y = dialogue["label"].values

            self.dialogues.append((
                torch.tensor(x_wav, dtype=torch.float32),
                torch.tensor(x_ac, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        return self.dialogues[idx]


def collate_fn(batch):

    xs_wav, xs_ac, ys = [], [], []

    for x_wav, x_ac, y in batch:
        xs_wav.append(x_wav)
        xs_ac.append(x_ac)
        ys.append(y)

    lengths = torch.tensor([len(x) for x in xs_wav])

    xs_wav = pad_sequence(xs_wav, batch_first=True)
    xs_ac = pad_sequence(xs_ac, batch_first=True)
    ys = pad_sequence(ys, batch_first=True, padding_value=-100)

    return xs_wav, xs_ac, ys, lengths


EMBED_DIR = config.EMBEDDINGS_DIR

OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "bilstm_fusion")
CHECKPOINT_DIR = os.path.join(config.CHECKPOINT_DIR, "bilstm_fusion")

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


# Load ALL-LAYER wav2vec2 embeddings (13, 768) per utterance

train_wav2vec = np.load(os.path.join(EMBED_DIR, "train_embeddings_layers.npy"))
dev_wav2vec = np.load(os.path.join(EMBED_DIR, "dev_embeddings_layers.npy"))

num_layers = train_wav2vec.shape[1]
wav2vec_dim = train_wav2vec.shape[2]

print("Train wav2vec2 layer embeddings:", train_wav2vec.shape)
print("Dev wav2vec2 layer embeddings:", dev_wav2vec.shape)


# Load acoustic flat-aggregate features (192,), already StandardScaler-
# normalized (fit on train only) by utils/extract_acoustic_features.py.

train_acoustic = np.load(os.path.join(EMBED_DIR, "train_acoustic_flat.npy"))
dev_acoustic = np.load(os.path.join(EMBED_DIR, "dev_acoustic_flat.npy"))

acoustic_dim = train_acoustic.shape[1]

print("Train acoustic features:", train_acoustic.shape)
print("Dev acoustic features:", dev_acoustic.shape)


# Normalize wav2vec2 embeddings (fit on train only). Fit a NEW scaler over
# the flattened (num_layers * hidden_dim) features -- do NOT reuse or
# overwrite data/embeddings/scaler.pkl. The acoustic block is already
# scaled at extraction time and is NOT rescaled here.

scaler = StandardScaler()

train_wav2vec_flat = train_wav2vec.reshape(train_wav2vec.shape[0], -1)
dev_wav2vec_flat = dev_wav2vec.reshape(dev_wav2vec.shape[0], -1)

train_wav2vec_flat = scaler.fit_transform(train_wav2vec_flat)
dev_wav2vec_flat = scaler.transform(dev_wav2vec_flat)

train_wav2vec = train_wav2vec_flat.reshape(-1, num_layers, wav2vec_dim)
dev_wav2vec = dev_wav2vec_flat.reshape(-1, num_layers, wav2vec_dim)

with open(os.path.join(OUTPUT_DIR, "scaler_wav2vec.pkl"), "wb") as file:
    pickle.dump(scaler, file)


train_meta = pd.read_csv(os.path.join(EMBED_DIR, "train_metadata.csv"))
dev_meta = pd.read_csv(os.path.join(EMBED_DIR, "dev_metadata.csv"))


# Label encoder (shared, fit on train only by train_mlp.py)

encoder = joblib.load(os.path.join(EMBED_DIR, "label_encoder.pkl"))

print("Classes:", encoder.classes_)


train_dataset = DialogueDataset(train_wav2vec, train_acoustic, train_meta, encoder)
dev_dataset = DialogueDataset(dev_wav2vec, dev_acoustic, dev_meta, encoder)


train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_fn
)


# Model

model = EmotionBiLSTMWithLayerWeightingFusion(
    wav2vec_dim=wav2vec_dim,
    acoustic_dim=acoustic_dim,
    hidden_dim=256,
    num_classes=len(encoder.classes_),
    num_layers=num_layers
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

best_path = os.path.join(CHECKPOINT_DIR, "bilstm_fusion_best.pt")


# Training

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for x_wav, x_ac, y, lengths in train_loader:

        x_wav = x_wav.to(device)
        x_ac = x_ac.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(x_wav, x_ac, lengths)

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

        for x_wav, x_ac, y, lengths in dev_loader:

            x_wav = x_wav.to(device)
            x_ac = x_ac.to(device)

            output = model(x_wav, x_ac, lengths)

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

    for x_wav, x_ac, y, lengths in dev_loader:

        x_wav = x_wav.to(device)
        x_ac = x_ac.to(device)

        output = model(x_wav, x_ac, lengths)

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

np.save(os.path.join(OUTPUT_DIR, "confusion_matrix.npy"), matrix)

prediction_df = pd.DataFrame(
    {
        "True Label": encoder.inverse_transform(ground_truth),
        "Predicted Label": encoder.inverse_transform(predictions)
    }
)

prediction_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

learned_layer_weights = model.layer_weighting.get_weights().detach().cpu().tolist()


results = {

    "model": "bilstm_fusion",

    "embedding_strategy": "wav2vec2_layer_weighted_concat_acoustic_flat_aggregate",

    "accuracy": accuracy_score(ground_truth, predictions),
    "macro_f1": f1_score(ground_truth, predictions, average="macro"),
    "weighted_f1": f1_score(ground_truth, predictions, average="weighted"),

    "best_macro_f1": best_f1,

    "num_training_samples": len(train_dataset),
    "num_validation_samples": len(dev_dataset),

    "epochs_run": epochs_run,
    "patience": patience,

    "wav2vec_dim": wav2vec_dim,
    "acoustic_dim": acoustic_dim,
    "fused_dim": wav2vec_dim + acoustic_dim,

    "learned_layer_weights": learned_layer_weights,

    "classes": list(encoder.classes_)

}

with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as file:
    json.dump(results, file, indent=4)

print("\nOutputs saved to:", OUTPUT_DIR)
print("✓ classification_report.txt")
print("✓ confusion_matrix.npy")
print("✓ predictions.csv")
print("✓ results.json")
print("✓ scaler_wav2vec.pkl")
