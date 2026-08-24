"""
Single, official test-set evaluation pass for the five Tier A (acoustic-only)
model variants: mlp, bilstm, gat, gat_context, gat_context_v2.

Mirrors evaluate_test_set_layerweighted.py's structure and scope exactly, but
loads test_acoustic_flat.npy / test_acoustic_frames.npy (produced by
utils/extract_acoustic_features.py, already scaled by its own train-fit
scalers) instead of wav2vec2 embeddings, and loads checkpoints from the
*_acoustic directories.

Does not touch dev, does not retrain anything. Runs once.
"""

import os
import json

import joblib
import numpy as np
import pandas as pd

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
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
from utils.frame_bilstm import FrameDialogueDataset, frame_collate_fn, EmotionBiLSTMWithFrameEncoder


EMBED_DIR = config.EMBEDDINGS_DIR
CHECKPOINT_DIR = config.CHECKPOINT_DIR
OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "test_evaluation_acoustic")

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using:", device)


# ---------------------------------------------------------------------------
# Shared test-set loading
# ---------------------------------------------------------------------------

test_acoustic_flat = np.load(os.path.join(EMBED_DIR, "test_acoustic_flat.npy"))
test_acoustic_frames = np.load(os.path.join(EMBED_DIR, "test_acoustic_frames.npy"), allow_pickle=True)
test_meta = pd.read_csv(os.path.join(EMBED_DIR, "test_metadata.csv"))

print("Test utterances:", len(test_meta))
print("Test acoustic flat features:", test_acoustic_flat.shape)

encoder = joblib.load(os.path.join(EMBED_DIR, "label_encoder.pkl"))
print("Label encoder classes:", list(encoder.classes_))

y_test = encoder.transform(test_meta["emotion"])


def full_report(y_true, y_pred, model_name):

    report_dict = classification_report(
        y_true, y_pred, target_names=encoder.classes_, zero_division=0, output_dict=True
    )

    report_text = classification_report(
        y_true, y_pred, target_names=encoder.classes_, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(encoder.classes_))))

    with open(os.path.join(OUTPUT_DIR, f"{model_name}_classification_report.txt"), "w") as f:
        f.write(report_text)

    np.save(os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix.npy"), cm)

    print(f"\n=== {model_name} (acoustic-only) — TEST SET ===")
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
# 1. MLP (utterance-level, flat aggregate)
# ---------------------------------------------------------------------------

mlp_model = EmotionMLP(input_dim=test_acoustic_flat.shape[1], num_classes=len(encoder.classes_))
mlp_model.load_state_dict(
    torch.load(os.path.join(CHECKPOINT_DIR, "mlp_acoustic", "mlp_acoustic_best.pt"), map_location=device)
)
mlp_model.to(device)
mlp_model.eval()

with torch.no_grad():
    x = torch.tensor(test_acoustic_flat, dtype=torch.float32).to(device)
    output = mlp_model(x)
    mlp_preds = torch.argmax(output, dim=1).cpu().numpy()

all_results.append(full_report(y_test, mlp_preds, "mlp"))


# ---------------------------------------------------------------------------
# 2. BiLSTM (dialogue-level, frame-encoder-pooled per utterance)
# ---------------------------------------------------------------------------

frame_dim = test_acoustic_frames[0].shape[1]

test_dataset = FrameDialogueDataset(test_acoustic_frames, test_meta, encoder)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=frame_collate_fn)

bilstm_model = EmotionBiLSTMWithFrameEncoder(
    bilstm_cls=EmotionBiLSTM,
    frame_input_dim=frame_dim,
    frame_hidden_dim=128,
    bilstm_hidden_dim=256,
    num_classes=len(encoder.classes_)
)
bilstm_model.load_state_dict(
    torch.load(os.path.join(CHECKPOINT_DIR, "bilstm_acoustic", "bilstm_acoustic_best.pt"), map_location=device)
)
bilstm_model.to(device)
bilstm_model.eval()

bilstm_true, bilstm_preds = [], []

with torch.no_grad():

    for padded_frames, frame_lengths, utt_lengths, y in test_loader:

        padded_frames = padded_frames.to(device)

        output = bilstm_model(padded_frames, frame_lengths, utt_lengths)
        pred = torch.argmax(output, dim=-1).cpu()

        mask = y != -100

        bilstm_preds.extend(pred[mask].numpy())
        bilstm_true.extend(y[mask].numpy())

all_results.append(full_report(bilstm_true, bilstm_preds, "bilstm"))


# ---------------------------------------------------------------------------
# 3-5. GAT variants (chain / context / context+DropEdge)
# ---------------------------------------------------------------------------

def evaluate_gat(build_graphs_fn, checkpoint_path, model_name):

    graphs, _ = build_graphs_fn(test_acoustic_flat, test_meta, encoder=encoder)

    model = EmotionGAT(input_dim=test_acoustic_flat.shape[1], num_classes=len(encoder.classes_))
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
        os.path.join(CHECKPOINT_DIR, "gat_acoustic", "gat_acoustic_best.pt"),
        "gat"
    )
)

all_results.append(
    evaluate_gat(
        build_context_graphs,
        os.path.join(CHECKPOINT_DIR, "gat_context_acoustic", "gat_context_acoustic_best.pt"),
        "gat_context"
    )
)

all_results.append(
    evaluate_gat(
        build_context_graphs,
        os.path.join(CHECKPOINT_DIR, "gat_context_v2_acoustic", "gat_context_v2_acoustic_best.pt"),
        "gat_context_v2"
    )
)


# ---------------------------------------------------------------------------
# Consolidated table
# ---------------------------------------------------------------------------

print("\n\n=== CONSOLIDATED TEST-SET RESULTS (five acoustic-only models) ===\n")

header = f"{'model':<16}{'accuracy':>10}{'macro_f1':>10}{'weighted_f1':>13}"
print(header)
print("-" * len(header))

for r in all_results:
    print(f"{r['model']:<16}{r['accuracy']:>10.4f}{r['macro_f1']:>10.4f}{r['weighted_f1']:>13.4f}")

with open(os.path.join(OUTPUT_DIR, "consolidated_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "consolidated_results.txt"), "w") as f:
    f.write(header + "\n")
    f.write("-" * len(header) + "\n")
    for r in all_results:
        f.write(f"{r['model']:<16}{r['accuracy']:>10.4f}{r['macro_f1']:>10.4f}{r['weighted_f1']:>13.4f}\n")

print(f"\nSaved: {OUTPUT_DIR}/consolidated_results.json")
print(f"Saved: {OUTPUT_DIR}/consolidated_results.txt")
print("Per-model classification reports and confusion matrices also saved in that directory.")
