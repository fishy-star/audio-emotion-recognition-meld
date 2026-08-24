import os
import json
import pickle
import random

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import TensorDataset, DataLoader

from models.mlp import EmotionMLP
from utils.embedding_strategies import TrainableLayerWeighting


class FocalLoss(nn.Module):

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):

        log_probs = torch.log_softmax(logits, dim=1)

        target_log_probs = log_probs.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)

        target_probs = target_log_probs.exp()

        loss = -((1 - target_probs) ** self.gamma) * target_log_probs

        if self.alpha is not None:
            loss = loss * self.alpha[targets]

        return loss.mean()


class EmotionMLPWithLayerWeightingFusion(nn.Module):
    """Concatenates the trainable softmax-weighted sum across all 13
    wav2vec2 layers with the (already-scaled) acoustic flat-aggregate
    vector, then feeds the fused representation into the unmodified
    EmotionMLP classifier."""

    def __init__(self, wav2vec_dim=768, acoustic_dim=192, num_classes=7, num_layers=13):
        super().__init__()

        self.layer_weighting = TrainableLayerWeighting(num_layers=num_layers)
        self.mlp = EmotionMLP(input_dim=wav2vec_dim + acoustic_dim, num_classes=num_classes)

    def forward(self, x_wav2vec, x_acoustic):
        # x_wav2vec: (batch, num_layers, hidden_dim), x_acoustic: (batch, acoustic_dim)
        pooled = self.layer_weighting(x_wav2vec)
        fused = torch.cat([pooled, x_acoustic], dim=-1)
        return self.mlp(fused)


EMBED_DIR = config.EMBEDDINGS_DIR

CHECKPOINT_DIR = os.path.join(config.CHECKPOINT_DIR, "mlp_fusion")
OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "mlp_fusion")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


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

X_train_wav2vec = np.load(os.path.join(EMBED_DIR, "train_embeddings_layers.npy"))
X_dev_wav2vec = np.load(os.path.join(EMBED_DIR, "dev_embeddings_layers.npy"))

num_layers = X_train_wav2vec.shape[1]
wav2vec_dim = X_train_wav2vec.shape[2]

print("Train wav2vec2 layer embeddings:", X_train_wav2vec.shape)
print("Dev wav2vec2 layer embeddings:", X_dev_wav2vec.shape)


# Load acoustic flat-aggregate features (192,), already StandardScaler-
# normalized (fit on train only) by utils/extract_acoustic_features.py.

X_train_acoustic = np.load(os.path.join(EMBED_DIR, "train_acoustic_flat.npy"))
X_dev_acoustic = np.load(os.path.join(EMBED_DIR, "dev_acoustic_flat.npy"))

acoustic_dim = X_train_acoustic.shape[1]

print("Train acoustic features:", X_train_acoustic.shape)
print("Dev acoustic features:", X_dev_acoustic.shape)


# Load metadata

train_meta = pd.read_csv(os.path.join(EMBED_DIR, "train_metadata.csv"))
dev_meta = pd.read_csv(os.path.join(EMBED_DIR, "dev_metadata.csv"))


# Normalize wav2vec2 embeddings (fit on train only). Fit a NEW scaler over
# the flattened (num_layers * hidden_dim) features -- do NOT reuse or
# overwrite data/embeddings/scaler.pkl, which was fit on the single
# last-layer mean-pooled embeddings and is shared with other scripts. The
# acoustic block is already scaled at extraction time, so it is NOT rescaled
# here -- only the wav2vec2 block gets its own scaler, same as train_mlp_v2.py.

scaler = StandardScaler()

X_train_wav2vec_flat = X_train_wav2vec.reshape(X_train_wav2vec.shape[0], -1)
X_dev_wav2vec_flat = X_dev_wav2vec.reshape(X_dev_wav2vec.shape[0], -1)

X_train_wav2vec_flat = scaler.fit_transform(X_train_wav2vec_flat)
X_dev_wav2vec_flat = scaler.transform(X_dev_wav2vec_flat)

X_train_wav2vec = X_train_wav2vec_flat.reshape(-1, num_layers, wav2vec_dim)
X_dev_wav2vec = X_dev_wav2vec_flat.reshape(-1, num_layers, wav2vec_dim)

with open(os.path.join(OUTPUT_DIR, "scaler_wav2vec.pkl"), "wb") as file:
    pickle.dump(scaler, file)


# Encode labels. Fit a NEW encoder and save it under outputs/mlp_fusion --
# do NOT overwrite the shared data/embeddings/label_encoder.pkl.

encoder = LabelEncoder()

y_train = encoder.fit_transform(train_meta["emotion"])
y_dev = encoder.transform(dev_meta["emotion"])

with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as file:
    pickle.dump(encoder, file)

print("Classes:")
print(encoder.classes_)


# Tensor conversion

X_train_wav2vec = torch.tensor(X_train_wav2vec, dtype=torch.float32)
X_train_acoustic = torch.tensor(X_train_acoustic, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_dev_wav2vec_tensor = torch.tensor(X_dev_wav2vec, dtype=torch.float32)
X_dev_acoustic_tensor = torch.tensor(X_dev_acoustic, dtype=torch.float32)


dataset = TensorDataset(X_train_wav2vec, X_train_acoustic, y_train_tensor)

loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Model

model = EmotionMLPWithLayerWeightingFusion(
    wav2vec_dim=wav2vec_dim,
    acoustic_dim=acoustic_dim,
    num_classes=len(encoder.classes_),
    num_layers=num_layers
)

model.to(device)


# Class weights

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

weights = torch.tensor(weights, dtype=torch.float32).to(device)


criterion = FocalLoss(alpha=weights, gamma=2.0)

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

best_path = os.path.join(CHECKPOINT_DIR, "mlp_fusion_best.pt")


# Training

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for X_wav_batch, X_ac_batch, y_batch in loader:

        X_wav_batch = X_wav_batch.to(device)
        X_ac_batch = X_ac_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        output = model(X_wav_batch, X_ac_batch)

        loss = criterion(output, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    # Validation

    model.eval()

    with torch.no_grad():

        output = model(
            X_dev_wav2vec_tensor.to(device),
            X_dev_acoustic_tensor.to(device)
        )

        predictions = torch.argmax(output, dim=1).cpu().numpy()


    accuracy = accuracy_score(y_dev, predictions)
    macro_f1 = f1_score(y_dev, predictions, average="macro")
    weighted_f1 = f1_score(y_dev, predictions, average="weighted")

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

with torch.no_grad():

    output = model(
        X_dev_wav2vec_tensor.to(device),
        X_dev_acoustic_tensor.to(device)
    )

    predictions = torch.argmax(output, dim=1).cpu().numpy()


print("\nBest Model Evaluation\n")

accuracy = accuracy_score(y_dev, predictions)
macro_f1 = f1_score(y_dev, predictions, average="macro")
weighted_f1 = f1_score(y_dev, predictions, average="weighted")

report = classification_report(
    y_dev,
    predictions,
    target_names=encoder.classes_,
    zero_division=0
)

print(report)

matrix = confusion_matrix(y_dev, predictions)

with open(
    os.path.join(OUTPUT_DIR, "classification_report.txt"),
    "w"
) as file:

    file.write(report)

np.save(os.path.join(OUTPUT_DIR, "confusion_matrix.npy"), matrix)

prediction_df = pd.DataFrame(
    {
        "True Label": encoder.inverse_transform(y_dev),
        "Predicted Label": encoder.inverse_transform(predictions)
    }
)

prediction_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

learned_layer_weights = model.layer_weighting.get_weights().detach().cpu().tolist()


results = {

    "model": "mlp_fusion",

    "embedding_strategy": "wav2vec2_layer_weighted_concat_acoustic_flat_aggregate",

    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "weighted_f1": weighted_f1,

    "best_macro_f1": best_f1,

    "num_training_samples": len(X_train_wav2vec),
    "num_validation_samples": len(X_dev_wav2vec_tensor),

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
print("✓ label_encoder.pkl")
print("✓ scaler_wav2vec.pkl")
