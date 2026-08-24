import os
import json
import random
import pickle

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

from torch.utils.data import WeightedRandomSampler

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from torch_geometric.data import Data

from models.gat import EmotionGAT
from utils.graph_context import build_graphs
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


class EmotionGATWithLayerWeightingFusion(nn.Module):
    """Concatenates the trainable softmax-weighted sum across all 13
    wav2vec2 layers with the (already-scaled) acoustic flat-aggregate
    node feature, per node, before the unmodified EmotionGAT."""

    def __init__(self, wav2vec_dim=768, acoustic_dim=192, num_classes=7, num_layers=13):
        super().__init__()

        self.layer_weighting = TrainableLayerWeighting(num_layers=num_layers)
        self.gat = EmotionGAT(input_dim=wav2vec_dim + acoustic_dim, num_classes=num_classes)

    def forward(self, x_wav2vec, x_acoustic, edge_index):
        # x_wav2vec: (num_nodes, num_layers, hidden_dim), x_acoustic: (num_nodes, acoustic_dim)
        pooled = self.layer_weighting(x_wav2vec)
        fused = torch.cat([pooled, x_acoustic], dim=-1)
        return self.gat(fused, edge_index)


EMBED_DIR = config.EMBEDDINGS_DIR

OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "gat_context_fusion")
CHECKPOINT_DIR = os.path.join(config.CHECKPOINT_DIR, "gat_context_fusion")

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


train_meta = pd.read_csv(os.path.join(EMBED_DIR, "train_metadata.csv"))
dev_meta = pd.read_csv(os.path.join(EMBED_DIR, "dev_metadata.csv"))


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


shared_encoder = joblib.load(os.path.join(EMBED_DIR, "label_encoder.pkl"))

print("Loaded shared label encoder classes:", shared_encoder.classes_)

# Two parallel graph builds over the SAME metadata (same groupby/sort_values
# ordering deterministically), one per feature source -- lets each dialogue's
# wav2vec2-layer graph and acoustic graph be zipped node-for-node below.

train_graphs_wav2vec, encoder = build_graphs(train_wav2vec, train_meta, encoder=shared_encoder)
train_graphs_acoustic, _ = build_graphs(train_acoustic, train_meta, encoder=encoder)

dev_graphs_wav2vec, _ = build_graphs(dev_wav2vec, dev_meta, encoder=encoder)
dev_graphs_acoustic, _ = build_graphs(dev_acoustic, dev_meta, encoder=encoder)


def to_data_list(graphs_wav2vec, graphs_acoustic):

    data_list = []

    for gw, ga in zip(graphs_wav2vec, graphs_acoustic):

        assert gw["dialogue_id"] == ga["dialogue_id"]

        data_list.append(
            Data(
                x=torch.tensor(gw["x"], dtype=torch.float32),
                x_acoustic=torch.tensor(ga["x"], dtype=torch.float32),
                edge_index=torch.tensor(gw["edge_index"], dtype=torch.long),
                y=torch.tensor(gw["y"], dtype=torch.long)
            )
        )

    return data_list


train_data = to_data_list(train_graphs_wav2vec, train_graphs_acoustic)
dev_data = to_data_list(dev_graphs_wav2vec, dev_graphs_acoustic)

print("Training Graphs:", len(train_data))
print("Validation Graphs:", len(dev_data))


# Oversampling: upweight dialogue-graphs containing disgust/fear utterances

minority_classes = encoder.transform(["disgust", "fear"])

graph_sample_weights = []

for graph in train_data:

    y = graph.y.numpy()

    if np.isin(y, minority_classes).any():
        graph_sample_weights.append(3.5)
    else:
        graph_sample_weights.append(1.0)

print(
    "Training graphs containing disgust/fear:",
    sum(1 for w in graph_sample_weights if w > 1.0),
    "/",
    len(train_data)
)

train_sampler = WeightedRandomSampler(
    weights=graph_sample_weights,
    num_samples=len(train_data),
    replacement=True
)


train_labels = encoder.transform(train_meta["emotion"])

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

weights = torch.tensor(weights, dtype=torch.float32).to(device)

print(encoder.classes_)
print(weights)


model = EmotionGATWithLayerWeightingFusion(
    wav2vec_dim=wav2vec_dim,
    acoustic_dim=acoustic_dim,
    num_classes=len(encoder.classes_),
    num_layers=num_layers
)

model.to(device)


criterion = FocalLoss(alpha=weights, gamma=2.0)

assert criterion.alpha is not None, "Class weights must reach FocalLoss"
print("Class weights reaching FocalLoss:", criterion.alpha)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)


epochs = 40

patience = 8

counter = 0

best_f1 = 0.0

best_path = os.path.join(CHECKPOINT_DIR, "gat_context_fusion_best.pt")

for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for idx in train_sampler:

        graph = train_data[idx].to(device)

        optimizer.zero_grad()

        output = model(graph.x, graph.x_acoustic, graph.edge_index)

        loss = criterion(output, graph.y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():

        for graph in dev_data:

            graph = graph.to(device)

            output = model(graph.x, graph.x_acoustic, graph.edge_index)

            pred = torch.argmax(output, dim=1)

            predictions.extend(pred.cpu().numpy())
            labels.extend(graph.y.cpu().numpy())

    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")
    weighted_f1 = f1_score(labels, predictions, average="weighted")

    avg_loss = total_loss / len(train_data)

    scheduler.step(macro_f1)

    print(
        f"Epoch {epoch + 1}/{epochs}",
        f"Loss {avg_loss:.4f}",
        f"Accuracy {accuracy:.4f}",
        f"Macro F1 {macro_f1:.4f}",
        f"Weighted F1 {weighted_f1:.4f}",
        f"LR {optimizer.param_groups[0]['lr']:.6f}"
    )

    if macro_f1 > best_f1:

        best_f1 = macro_f1
        counter = 0

        torch.save(model.state_dict(), best_path)

        print("\nBest Model Saved\n")

    else:

        counter += 1

        print(f"No improvement ({counter}/{patience})")

        if counter >= patience:

            print("\nEarly stopping triggered.\n")
            break

epochs_run = epoch + 1

model.load_state_dict(torch.load(best_path, map_location=device))
model.eval()

predictions = []
labels = []

with torch.no_grad():

    for graph in dev_data:

        graph = graph.to(device)

        output = model(graph.x, graph.x_acoustic, graph.edge_index)

        pred = torch.argmax(output, dim=1)

        predictions.extend(pred.cpu().numpy())
        labels.extend(graph.y.cpu().numpy())

print("\nBest Model Evaluation\n")

report = classification_report(
    labels,
    predictions,
    target_names=encoder.classes_,
    zero_division=0
)

print(report)

matrix = confusion_matrix(labels, predictions)

with open(
    os.path.join(OUTPUT_DIR, "classification_report.txt"),
    "w"
) as file:

    file.write(report)

np.save(os.path.join(OUTPUT_DIR, "confusion_matrix.npy"), matrix)

prediction_df = pd.DataFrame(
    {
        "True Label": encoder.inverse_transform(labels),
        "Predicted Label": encoder.inverse_transform(predictions)
    }
)

prediction_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as file:
    pickle.dump(encoder, file)

learned_layer_weights = model.layer_weighting.get_weights().detach().cpu().tolist()


results = {

    "model": "gat_context_fusion",

    "embedding_strategy": "wav2vec2_layer_weighted_concat_acoustic_flat_aggregate",

    "accuracy": accuracy_score(labels, predictions),
    "macro_f1": f1_score(labels, predictions, average="macro"),
    "weighted_f1": f1_score(labels, predictions, average="weighted"),

    "num_training_samples": len(train_data),
    "num_validation_samples": len(dev_data),

    "epochs_run": epochs_run,
    "best_macro_f1": best_f1,
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
