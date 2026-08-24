"""
Single, official test-set evaluation pass for the five Tier B (fusion:
wav2vec2 layer-weighted + acoustic flat-aggregate) model variants: mlp,
bilstm, gat, gat_context, gat_context_v2.

Mirrors evaluate_test_set_layerweighted.py's structure and scope exactly,
but loads BOTH test_embeddings_layers.npy (13, 768) and test_acoustic_flat.npy
(already scaled by its own train-fit scaler), normalizes the wav2vec2 block
with each model's own scaler_wav2vec.pkl (fit independently per model on the
flattened 13*768 train features, same convention as scaler_v2.pkl in the
layer-weighted tier), and loads checkpoints from the *_fusion directories.

Does not touch dev, does not retrain anything. Runs once.
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


class EmotionMLPWithLayerWeightingFusion(nn.Module):

    def __init__(self, wav2vec_dim=768, acoustic_dim=192, num_classes=7, num_layers=13):
        super().__init__()
        self.layer_weighting = TrainableLayerWeighting(num_layers=num_layers)
        self.mlp = EmotionMLP(input_dim=wav2vec_dim + acoustic_dim, num_classes=num_classes)

    def forward(self, x_wav2vec, x_acoustic):
        pooled = self.layer_weighting(x_wav2vec)
        return self.mlp(torch.cat([pooled, x_acoustic], dim=-1))


class EmotionBiLSTMWithLayerWeightingFusion(nn.Module):

    def __init__(self, wav2vec_dim=768, acoustic_dim=192, hidden_dim=256, num_classes=7, num_layers=13):
        super().__init__()
        self.layer_weighting = TrainableLayerWeighting(num_layers=num_layers)
        self.bilstm = EmotionBiLSTM(input_dim=wav2vec_dim + acoustic_dim, hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, x_wav2vec, x_acoustic, lengths):
        pooled = self.layer_weighting(x_wav2vec)
        return self.bilstm(torch.cat([pooled, x_acoustic], dim=-1), lengths)


class EmotionGATWithLayerWeightingFusion(nn.Module):

    def __init__(self, wav2vec_dim=768, acoustic_dim=192, num_classes=7, num_layers=13):
        super().__init__()
        self.layer_weighting = TrainableLayerWeighting(num_layers=num_layers)
        self.gat = EmotionGAT(input_dim=wav2vec_dim + acoustic_dim, num_classes=num_classes)

    def forward(self, x_wav2vec, x_acoustic, edge_index):
        pooled = self.layer_weighting(x_wav2vec)
        return self.gat(torch.cat([pooled, x_acoustic], dim=-1), edge_index)


EMBED_DIR = config.EMBEDDINGS_DIR
CHECKPOINT_DIR = config.CHECKPOINT_DIR
OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "test_evaluation_fusion")

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

test_wav2vec_raw = np.load(os.path.join(EMBED_DIR, "test_embeddings_layers.npy"))
test_acoustic = np.load(os.path.join(EMBED_DIR, "test_acoustic_flat.npy"))
test_meta = pd.read_csv(os.path.join(EMBED_DIR, "test_metadata.csv"))

num_layers = test_wav2vec_raw.shape[1]
wav2vec_dim = test_wav2vec_raw.shape[2]
acoustic_dim = test_acoustic.shape[1]

print("Test utterances:", len(test_meta))
print("Test wav2vec2 layer embeddings:", test_wav2vec_raw.shape)
print("Test acoustic flat features:", test_acoustic.shape)

encoder = joblib.load(os.path.join(EMBED_DIR, "label_encoder.pkl"))
print("Label encoder classes:", list(encoder.classes_))

y_test = encoder.transform(test_meta["emotion"])


def scale_wav2vec(scaler_path, embeddings):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    flat = embeddings.reshape(embeddings.shape[0], -1)
    flat = scaler.transform(flat)

    return flat.reshape(-1, num_layers, wav2vec_dim)


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

    print(f"\n=== {model_name} (fusion) — TEST SET ===")
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
# 1. MLP fusion (utterance-level)
# ---------------------------------------------------------------------------

mlp_test_wav2vec = scale_wav2vec(
    os.path.join(config.OUTPUT_DIR, "mlp_fusion", "scaler_wav2vec.pkl"),
    test_wav2vec_raw
)

mlp_model = EmotionMLPWithLayerWeightingFusion(
    wav2vec_dim=wav2vec_dim, acoustic_dim=acoustic_dim, num_classes=len(encoder.classes_), num_layers=num_layers
)
mlp_model.load_state_dict(
    torch.load(os.path.join(CHECKPOINT_DIR, "mlp_fusion", "mlp_fusion_best.pt"), map_location=device)
)
mlp_model.to(device)
mlp_model.eval()

with torch.no_grad():
    x_wav = torch.tensor(mlp_test_wav2vec, dtype=torch.float32).to(device)
    x_ac = torch.tensor(test_acoustic, dtype=torch.float32).to(device)
    output = mlp_model(x_wav, x_ac)
    mlp_preds = torch.argmax(output, dim=1).cpu().numpy()

all_results.append(full_report(y_test, mlp_preds, "mlp"))


# ---------------------------------------------------------------------------
# 2. BiLSTM fusion (dialogue-level, padded sequences)
# ---------------------------------------------------------------------------

bilstm_test_wav2vec = scale_wav2vec(
    os.path.join(config.OUTPUT_DIR, "bilstm_fusion", "scaler_wav2vec.pkl"),
    test_wav2vec_raw
)

test_meta_labeled = test_meta.copy()
test_meta_labeled["label"] = y_test

bilstm_dialogues = []

for _, dialogue in test_meta_labeled.groupby("dialogue_id", sort=False):

    dialogue = dialogue.sort_values("utterance_id")
    indices = dialogue.index.tolist()

    x_wav = bilstm_test_wav2vec[indices]
    x_ac = test_acoustic[indices]
    y = dialogue["label"].values

    bilstm_dialogues.append((
        torch.tensor(x_wav, dtype=torch.float32),
        torch.tensor(x_ac, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    ))

bilstm_model = EmotionBiLSTMWithLayerWeightingFusion(
    wav2vec_dim=wav2vec_dim, acoustic_dim=acoustic_dim, hidden_dim=256,
    num_classes=len(encoder.classes_), num_layers=num_layers
)
bilstm_model.load_state_dict(
    torch.load(os.path.join(CHECKPOINT_DIR, "bilstm_fusion", "bilstm_fusion_best.pt"), map_location=device)
)
bilstm_model.to(device)
bilstm_model.eval()

bilstm_true, bilstm_preds = [], []

with torch.no_grad():

    for i in range(0, len(bilstm_dialogues), 8):

        batch = bilstm_dialogues[i:i + 8]

        xs_wav = [b[0] for b in batch]
        xs_ac = [b[1] for b in batch]
        ys = [b[2] for b in batch]

        lengths = torch.tensor([len(x) for x in xs_wav])

        xs_wav_padded = pad_sequence(xs_wav, batch_first=True).to(device)
        xs_ac_padded = pad_sequence(xs_ac, batch_first=True).to(device)
        ys_padded = pad_sequence(ys, batch_first=True, padding_value=-100)

        output = bilstm_model(xs_wav_padded, xs_ac_padded, lengths)
        pred = torch.argmax(output, dim=-1).cpu()

        mask = ys_padded != -100

        bilstm_preds.extend(pred[mask].numpy())
        bilstm_true.extend(ys_padded[mask].numpy())

all_results.append(full_report(bilstm_true, bilstm_preds, "bilstm"))


# ---------------------------------------------------------------------------
# 3-5. GAT variants fusion (chain / context / context+DropEdge)
# ---------------------------------------------------------------------------

def evaluate_gat_fusion(build_graphs_fn, scaler_path, checkpoint_path, model_name):

    test_wav2vec = scale_wav2vec(scaler_path, test_wav2vec_raw)

    graphs_wav2vec, _ = build_graphs_fn(test_wav2vec, test_meta, encoder=encoder)
    graphs_acoustic, _ = build_graphs_fn(test_acoustic, test_meta, encoder=encoder)

    model = EmotionGATWithLayerWeightingFusion(
        wav2vec_dim=wav2vec_dim, acoustic_dim=acoustic_dim, num_classes=len(encoder.classes_), num_layers=num_layers
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    preds, labels = [], []

    with torch.no_grad():

        for gw, ga in zip(graphs_wav2vec, graphs_acoustic):

            assert gw["dialogue_id"] == ga["dialogue_id"]

            data = Data(
                x=torch.tensor(gw["x"], dtype=torch.float32),
                x_acoustic=torch.tensor(ga["x"], dtype=torch.float32),
                edge_index=torch.tensor(gw["edge_index"], dtype=torch.long),
                y=torch.tensor(gw["y"], dtype=torch.long)
            ).to(device)

            output = model(data.x, data.x_acoustic, data.edge_index)
            pred = torch.argmax(output, dim=1)

            preds.extend(pred.cpu().numpy())
            labels.extend(data.y.cpu().numpy())

    return full_report(labels, preds, model_name)


all_results.append(
    evaluate_gat_fusion(
        build_chain_graphs,
        os.path.join(config.OUTPUT_DIR, "gat_fusion", "scaler_wav2vec.pkl"),
        os.path.join(CHECKPOINT_DIR, "gat_fusion", "gat_fusion_best.pt"),
        "gat"
    )
)

all_results.append(
    evaluate_gat_fusion(
        build_context_graphs,
        os.path.join(config.OUTPUT_DIR, "gat_context_fusion", "scaler_wav2vec.pkl"),
        os.path.join(CHECKPOINT_DIR, "gat_context_fusion", "gat_context_fusion_best.pt"),
        "gat_context"
    )
)

all_results.append(
    evaluate_gat_fusion(
        build_context_graphs,
        os.path.join(config.OUTPUT_DIR, "gat_context_v2_fusion", "scaler_wav2vec.pkl"),
        os.path.join(CHECKPOINT_DIR, "gat_context_v2_fusion", "gat_context_v2_fusion_best.pt"),
        "gat_context_v2"
    )
)


# ---------------------------------------------------------------------------
# Consolidated table
# ---------------------------------------------------------------------------

print("\n\n=== CONSOLIDATED TEST-SET RESULTS (five fusion models) ===\n")

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
