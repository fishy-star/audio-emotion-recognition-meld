"""
Single, official test-set evaluation pass for the five layer-weighted
(TrainableLayerWeighting) model variants: mlp, bilstm, gat, gat_context,
gat_context_v2.

Mirrors evaluate_test_set.py's structure and scope exactly, but:
  - loads test_embeddings_layers.npy (13, 768) instead of test_embeddings.npy
  - normalizes with each model's own scaler_v2.pkl (fit independently per
    model on the flattened 13*768 train features), not the shared scaler.pkl
  - loads checkpoints from the *_v2 / *_lw directories, not the five
    original baseline checkpoints

Does not touch dev, does not retrain anything, does not touch the original
five baseline checkpoints/results. Runs once.
"""

import os
import json
import pickle

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from models.mlp import EmotionMLP
from models.bilstm import EmotionBiLSTM
from models.gat import EmotionGAT
from utils.graph import build_graphs as build_chain_graphs
from utils.graph_context import build_graphs as build_context_graphs
from utils.embedding_strategies import TrainableLayerWeighting


class EmotionMLPWithLayerWeighting(nn.Module):

    def __init__(self, input_dim=768, num_classes=7, num_layers=13):
        super().__init__()
        self.layer_weighting = TrainableLayerWeighting(num_layers=num_layers)
        self.mlp = EmotionMLP(input_dim=input_dim, num_classes=num_classes)

    def forward(self, x):
        return self.mlp(self.layer_weighting(x))


class EmotionBiLSTMWithLayerWeighting(nn.Module):

    def __init__(self, input_dim=768, hidden_dim=256, num_classes=7, num_layers=13):
        super().__init__()
        self.layer_weighting = TrainableLayerWeighting(num_layers=num_layers)
        self.bilstm = EmotionBiLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )

    def forward(self, x, lengths):
        return self.bilstm(self.layer_weighting(x), lengths)


class EmotionGATWithLayerWeighting(nn.Module):

    def __init__(self, input_dim=768, num_classes=7, num_layers=13):
        super().__init__()
        self.layer_weighting = TrainableLayerWeighting(num_layers=num_layers)
        self.gat = EmotionGAT(input_dim=input_dim, num_classes=num_classes)

    def forward(self, x, edge_index):
        return self.gat(self.layer_weighting(x), edge_index)


EMBED_DIR = config.EMBEDDINGS_DIR
CHECKPOINT_DIR = config.CHECKPOINT_DIR
OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "test_evaluation_layerweighted")

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using:", device)


# ---------------------------------------------------------------------------
# Shared test-set loading (13-layer embeddings + label encoder, read only)
# ---------------------------------------------------------------------------

test_embeddings_layers = np.load(
    os.path.join(EMBED_DIR, "test_embeddings_layers.npy")
)
test_meta = pd.read_csv(os.path.join(EMBED_DIR, "test_metadata.csv"))

num_layers = test_embeddings_layers.shape[1]
hidden_dim = test_embeddings_layers.shape[2]

print("Test utterances:", len(test_meta))
print("Test layer embeddings:", test_embeddings_layers.shape)
print("Test metadata columns:", list(test_meta.columns))

if "speaker" not in test_meta.columns:
    print(
        "\nNOTE: test_metadata.csv has no 'speaker' column. graph_context.py "
        "falls back to speaker=None in that case, so gat_context and "
        "gat_context_v2 would evaluate on test dialogues WITHOUT "
        "same-speaker edges. (Not applicable here if the column is present "
        "-- checked below.)\n"
    )

encoder = joblib.load(os.path.join(EMBED_DIR, "label_encoder.pkl"))
print("Label encoder classes:", list(encoder.classes_))

y_test = encoder.transform(test_meta["emotion"])


def scale_with(scaler_path, embeddings):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    flat = embeddings.reshape(embeddings.shape[0], -1)
    flat = scaler.transform(flat)

    return flat.reshape(-1, num_layers, hidden_dim)


def full_report(y_true, y_pred, model_name):

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=encoder.classes_,
        zero_division=0,
        output_dict=True
    )

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=encoder.classes_,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(encoder.classes_)))
    )

    with open(
        os.path.join(OUTPUT_DIR, f"{model_name}_classification_report.txt"),
        "w"
    ) as f:
        f.write(report_text)

    np.save(
        os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix.npy"),
        cm
    )

    print(f"\n=== {model_name} (layer-weighted) — TEST SET ===")
    print(report_text)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "per_class": {
            cls: {
                "precision": report_dict[cls]["precision"],
                "recall": report_dict[cls]["recall"],
                "f1": report_dict[cls]["f1-score"],
                "support": report_dict[cls]["support"]
            }
            for cls in encoder.classes_
        }
    }


all_results = []


# ---------------------------------------------------------------------------
# 1. MLP (utterance-level) — checkpoints/mlp_v2_epoch40/ (the patience=8
#    validated run; checkpoints/mlp_v2/ is the earlier un-validated 10-epoch
#    run and is deliberately NOT used here)
# ---------------------------------------------------------------------------

mlp_test_x = scale_with(
    os.path.join(config.OUTPUT_DIR, "mlp_v2_epoch40", "scaler_v2.pkl"),
    test_embeddings_layers
)

mlp_model = EmotionMLPWithLayerWeighting(
    input_dim=hidden_dim, num_classes=len(encoder.classes_), num_layers=num_layers
)
mlp_model.load_state_dict(
    torch.load(
        os.path.join(CHECKPOINT_DIR, "mlp_v2_epoch40", "mlp_v2_epoch40_best.pt"),
        map_location=device
    )
)
mlp_model.to(device)
mlp_model.eval()

with torch.no_grad():
    x = torch.tensor(mlp_test_x, dtype=torch.float32).to(device)
    output = mlp_model(x)
    mlp_preds = torch.argmax(output, dim=1).cpu().numpy()

all_results.append(full_report(y_test, mlp_preds, "mlp"))


# ---------------------------------------------------------------------------
# 2. BiLSTM (dialogue-level, padded sequences) — checkpoints/bilstm_v2/
# ---------------------------------------------------------------------------

bilstm_test_x = scale_with(
    os.path.join(config.OUTPUT_DIR, "bilstm_v2", "scaler_v2.pkl"),
    test_embeddings_layers
)

test_meta_labeled = test_meta.copy()
test_meta_labeled["label"] = y_test

bilstm_dialogues = []

for _, dialogue in test_meta_labeled.groupby("dialogue_id", sort=False):

    dialogue = dialogue.sort_values("utterance_id")
    indices = dialogue.index.tolist()

    x = bilstm_test_x[indices]
    y = dialogue["label"].values

    bilstm_dialogues.append(
        (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )
    )

bilstm_model = EmotionBiLSTMWithLayerWeighting(
    input_dim=hidden_dim,
    hidden_dim=256,
    num_classes=len(encoder.classes_),
    num_layers=num_layers
)
bilstm_model.load_state_dict(
    torch.load(
        os.path.join(CHECKPOINT_DIR, "bilstm_v2", "bilstm_v2_best.pt"),
        map_location=device
    )
)
bilstm_model.to(device)
bilstm_model.eval()

bilstm_true, bilstm_preds = [], []

with torch.no_grad():

    for i in range(0, len(bilstm_dialogues), 8):

        batch = bilstm_dialogues[i:i + 8]

        xs = [b[0] for b in batch]
        ys = [b[1] for b in batch]

        lengths = torch.tensor([len(x) for x in xs])

        xs_padded = pad_sequence(xs, batch_first=True).to(device)
        ys_padded = pad_sequence(ys, batch_first=True, padding_value=-100)

        output = bilstm_model(xs_padded, lengths)
        pred = torch.argmax(output, dim=-1).cpu()

        mask = ys_padded != -100

        bilstm_preds.extend(pred[mask].numpy())
        bilstm_true.extend(ys_padded[mask].numpy())

all_results.append(full_report(bilstm_true, bilstm_preds, "bilstm"))


# ---------------------------------------------------------------------------
# 3-5. GAT variants (chain / context / context+DropEdge) — DropEdge is
#      train-only, eval always uses the full graph, same as baseline
# ---------------------------------------------------------------------------

def evaluate_gat(build_graphs_fn, scaler_path, checkpoint_path, model_name):

    test_x = scale_with(scaler_path, test_embeddings_layers)

    graphs, _ = build_graphs_fn(test_x, test_meta, encoder=encoder)

    model = EmotionGATWithLayerWeighting(
        input_dim=hidden_dim, num_classes=len(encoder.classes_), num_layers=num_layers
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    preds, labels = [], []

    with torch.no_grad():

        for g in graphs:

            data = Data(
                x=torch.tensor(g["x"], dtype=torch.float32),
                edge_index=torch.tensor(g["edge_index"], dtype=torch.long),
                y=torch.tensor(g["y"], dtype=torch.long)
            ).to(device)

            output = model(data.x, data.edge_index)
            pred = torch.argmax(output, dim=1)

            preds.extend(pred.cpu().numpy())
            labels.extend(data.y.cpu().numpy())

    return full_report(labels, preds, model_name)


all_results.append(
    evaluate_gat(
        build_chain_graphs,
        os.path.join(config.OUTPUT_DIR, "gat_v2", "scaler_v2.pkl"),
        os.path.join(CHECKPOINT_DIR, "gat_v2", "gat_v2_best.pt"),
        "gat"
    )
)

all_results.append(
    evaluate_gat(
        build_context_graphs,
        os.path.join(config.OUTPUT_DIR, "gat_context_lw", "scaler_v2.pkl"),
        os.path.join(CHECKPOINT_DIR, "gat_context_lw", "gat_context_lw_best.pt"),
        "gat_context"
    )
)

all_results.append(
    evaluate_gat(
        build_context_graphs,
        os.path.join(config.OUTPUT_DIR, "gat_context_v2_lw", "scaler_v2.pkl"),
        os.path.join(CHECKPOINT_DIR, "gat_context_v2_lw", "gat_context_v2_lw_best.pt"),
        "gat_context_v2"
    )
)


# ---------------------------------------------------------------------------
# Consolidated table
# ---------------------------------------------------------------------------

print("\n\n=== CONSOLIDATED TEST-SET RESULTS (five layer-weighted models) ===\n")

header = f"{'model':<16}{'accuracy':>10}{'macro_f1':>10}{'weighted_f1':>13}"
print(header)
print("-" * len(header))

for r in all_results:
    print(
        f"{r['model']:<16}"
        f"{r['accuracy']:>10.4f}"
        f"{r['macro_f1']:>10.4f}"
        f"{r['weighted_f1']:>13.4f}"
    )

with open(os.path.join(OUTPUT_DIR, "consolidated_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "consolidated_results.txt"), "w") as f:
    f.write(header + "\n")
    f.write("-" * len(header) + "\n")
    for r in all_results:
        f.write(
            f"{r['model']:<16}"
            f"{r['accuracy']:>10.4f}"
            f"{r['macro_f1']:>10.4f}"
            f"{r['weighted_f1']:>13.4f}\n"
        )

print(f"\nSaved: {OUTPUT_DIR}/consolidated_results.json")
print(f"Saved: {OUTPUT_DIR}/consolidated_results.txt")
print("Per-model classification reports and confusion matrices also saved in that directory.")
