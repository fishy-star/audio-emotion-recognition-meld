"""
Step 3 of the speaker-edge bug fix: re-evaluate ONLY gat_context and
gat_context_v2 against their newly retrained (speaker-aware) checkpoints.

Does NOT touch mlp/bilstm/gat's test results — those remain final, from the
original evaluate_test_set.py run. This script only overwrites the
gat_context / gat_context_v2 entries in outputs/test_evaluation/.

This is the final, official number for gat_context and gat_context_v2. No
further retraining or re-evaluation of these two after this script runs.
"""

import os
import json
import pickle

import joblib
import numpy as np
import pandas as pd

import torch
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

from models.gat import EmotionGAT
from utils.graph_context import build_graphs as build_context_graphs


EMBED_DIR = config.EMBEDDINGS_DIR
CHECKPOINT_DIR = config.CHECKPOINT_DIR
OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "test_evaluation")

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("Using:", device)

test_embeddings_raw = np.load(os.path.join(EMBED_DIR, "test_embeddings.npy"))
test_meta = pd.read_csv(os.path.join(EMBED_DIR, "test_metadata.csv"))

with open(os.path.join(EMBED_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

test_embeddings = scaler.transform(test_embeddings_raw)

encoder = joblib.load(os.path.join(EMBED_DIR, "label_encoder.pkl"))
y_test = encoder.transform(test_meta["emotion"])

print("Test utterances:", len(test_meta))
print("speaker column present:", "speaker" in test_meta.columns)


def full_report(y_true, y_pred, model_name):

    report_dict = classification_report(
        y_true, y_pred, target_names=encoder.classes_,
        zero_division=0, output_dict=True
    )
    report_text = classification_report(
        y_true, y_pred, target_names=encoder.classes_, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(encoder.classes_))))

    with open(os.path.join(OUTPUT_DIR, f"{model_name}_classification_report.txt"), "w") as f:
        f.write(report_text)

    np.save(os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix.npy"), cm)

    print(f"\n=== {model_name} — TEST SET (corrected, speaker-aware retrain) ===")
    print(report_text)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    return {
        "model": model_name,
        "run_label": "corrected_speaker_aware_retrain",
        "note": "Final, official result. Retrained after fixing utils/graph_context.py's same-speaker edge loop range bug (was a strict subset of the window range, so it never added an edge). This is the only run of the two graph_context models that has genuine same-speaker edges.",
        "status": "final",
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


def evaluate(checkpoint_path, model_name):

    graphs, _ = build_context_graphs(test_embeddings, test_meta, encoder=encoder)

    model = EmotionGAT(input_dim=768, num_classes=len(encoder.classes_))
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


new_gat_context = evaluate(
    os.path.join(CHECKPOINT_DIR, "gat_context", "gat_best.pt"),
    "gat_context"
)

new_gat_context_v2 = evaluate(
    os.path.join(CHECKPOINT_DIR, "gat_context_v2", "gat_best.pt"),
    "gat_context_v2"
)


# ---------------------------------------------------------------------------
# Merge into consolidated_results.json, preserving mlp/bilstm/gat untouched,
# and keeping all three historical gat_context/gat_context_v2 result sets
# visible for the record.
# ---------------------------------------------------------------------------

consolidated_path = os.path.join(OUTPUT_DIR, "consolidated_results.json")

with open(consolidated_path) as f:
    existing = json.load(f)

# Historical entries: runs 1 (original window-only) and 2 (no-op speaker-fix
# re-run) were byte-identical — both reflect the OLD, buggy checkpoints, just
# under different test_metadata.csv states. Pull their numbers from the
# preserved pre-fix results (archived) rather than re-deriving them.
with open("archive/data_backups/test_evaluation_pre_speaker_fix/consolidated_results.json") as f:
    pre_fix = json.load(f)

pre_fix_by_model = {r["model"]: r for r in pre_fix}

historical_notes = {
    1: "Original run, before test_metadata.csv had a speaker column at all. Uses the OLD (buggy) graph_context.py, whose same-speaker edge loop never added an edge regardless — graphs were plain +/-2 window only.",
    2: "Re-run after fixing the missing speaker column on test_metadata.csv, but BEFORE fixing the graph_context.py loop-range bug. Byte-identical to run 1's numbers, which is exactly what revealed the deeper bug: the speaker column's presence made no difference because the edge logic never used it correctly."
}

merged = []

for entry in existing:

    if entry["model"] not in ("gat_context", "gat_context_v2"):
        # mlp, bilstm, gat (chain) — untouched, final, carried forward as-is
        merged.append(entry)
        continue

    model_name = entry["model"]
    base = pre_fix_by_model[model_name]

    for run_num in (1, 2):
        historical_entry = dict(base)
        historical_entry["run_label"] = (
            "original_window_only" if run_num == 1 else "no_op_speaker_fix_rerun"
        )
        historical_entry["note"] = historical_notes[run_num]
        historical_entry["status"] = "historical"
        merged.append(historical_entry)

# the two corrected retrains (run 3) go last, clearly marked final
merged.append(new_gat_context)
merged.append(new_gat_context_v2)

with open(consolidated_path, "w") as f:
    json.dump(merged, f, indent=2)

# human-readable table, same style as before but now showing all runs
table_path = os.path.join(OUTPUT_DIR, "consolidated_results.txt")

header = f"{'model':<16}{'run_label':<28}{'status':<12}{'accuracy':>10}{'macro_f1':>10}{'weighted_f1':>13}"

with open(table_path, "w") as f:
    f.write(header + "\n")
    f.write("-" * len(header) + "\n")
    for r in merged:
        run_label = r.get("run_label", "final")
        status = r.get("status", "final")
        f.write(
            f"{r['model']:<16}{run_label:<28}{status:<12}"
            f"{r['accuracy']:>10.4f}{r['macro_f1']:>10.4f}{r['weighted_f1']:>13.4f}\n"
        )

print("\n\n=== CONSOLIDATED TEST-SET RESULTS (mlp/bilstm/gat final; gat_context/gat_context_v2 all 3 runs) ===\n")
with open(table_path) as f:
    print(f.read())

print(f"Saved: {consolidated_path}")
print(f"Saved: {table_path}")
