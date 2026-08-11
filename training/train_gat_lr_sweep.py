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

from sklearn.utils.class_weight import (
    compute_class_weight
)

from torch_geometric.data import Data

from models.gat import EmotionGAT
from utils.graph import build_graphs


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


OUTPUT_DIR = os.path.join(
    config.OUTPUT_DIR,
    "gat"
)


CHECKPOINT_DIR = os.path.join(
    config.CHECKPOINT_DIR,
    "gat"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Using:",
    device
)


train_embeddings = np.load(
    os.path.join(
        EMBED_DIR,
        "train_embeddings.npy"
    )
)


dev_embeddings = np.load(
    os.path.join(
        EMBED_DIR,
        "dev_embeddings.npy"
    )
)


# Normalize embeddings using the scaler fit on train (from train_mlp.py)

with open(
    os.path.join(
        EMBED_DIR,
        "scaler.pkl"
    ),
    "rb"
) as file:

    scaler = pickle.load(file)

train_embeddings = scaler.transform(train_embeddings)
dev_embeddings = scaler.transform(dev_embeddings)


train_meta = pd.read_csv(
    os.path.join(
        EMBED_DIR,
        "train_metadata.csv"
    )
)


dev_meta = pd.read_csv(
    os.path.join(
        EMBED_DIR,
        "dev_metadata.csv"
    )
)


shared_encoder = joblib.load(
    os.path.join(
        EMBED_DIR,
        "label_encoder.pkl"
    )
)

print("Loaded shared label encoder classes:", shared_encoder.classes_)

train_graphs, encoder = build_graphs(
    train_embeddings,
    train_meta,
    encoder=shared_encoder
)


dev_graphs, _ = build_graphs(
    dev_embeddings,
    dev_meta,
    encoder=encoder
)


train_data = []

for graph in train_graphs:

    train_data.append(

        Data(

            x=torch.tensor(
                graph["x"],
                dtype=torch.float32
            ),

            edge_index=torch.tensor(
                graph["edge_index"],
                dtype=torch.long
            ),

            y=torch.tensor(
                graph["y"],
                dtype=torch.long
            )

        )

    )


dev_data = []

for graph in dev_graphs:

    dev_data.append(

        Data(

            x=torch.tensor(
                graph["x"],
                dtype=torch.float32
            ),

            edge_index=torch.tensor(
                graph["edge_index"],
                dtype=torch.long
            ),

            y=torch.tensor(
                graph["y"],
                dtype=torch.long
            )

        )

    )


print(
    "Training Graphs:",
    len(train_data)
)

print(
    "Validation Graphs:",
    len(dev_data)
)


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


train_labels = encoder.transform(
    train_meta["emotion"]
)


weights = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(
        train_labels
    ),

    y=train_labels

)


weights = torch.tensor(

    weights,

    dtype=torch.float32

).to(device)
print(
    encoder.classes_
)

print(
    weights
)




def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            pass


SEED = 42

LR_GRID = [5e-4, 1e-4, 5e-5, 1e-5]

EPOCHS = 100

PATIENCE = 8

sweep_results = []


for lr in LR_GRID:

    print(
        f"\n{'=' * 70}\nStarting run: lr={lr}\n{'=' * 70}\n"
    )

    set_seed(SEED)

    model = EmotionGAT(

        input_dim=768,

        num_classes=len(
            encoder.classes_
        )

    )

    model.to(device)

    criterion = FocalLoss(
        alpha=weights,
        gamma=2.0
    )

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=lr,

        weight_decay=1e-5

    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=3

    )

    train_sampler = WeightedRandomSampler(
        weights=graph_sample_weights,
        num_samples=len(train_data),
        replacement=True
    )

    counter = 0

    best_f1 = 0.0
    best_weighted_f1 = 0.0
    best_epoch = 0

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"gat_lr_sweep_{lr:.0e}.pt"
    )

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for idx in train_sampler:

            graph = train_data[idx]

            graph = graph.to(device)

            optimizer.zero_grad()

            output = model(
                graph.x,
                graph.edge_index
            )

            loss = criterion(
                output,
                graph.y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

        model.eval()

        predictions = []
        labels = []

        with torch.no_grad():

            for graph in dev_data:

                graph = graph.to(device)

                output = model(
                    graph.x,
                    graph.edge_index
                )

                pred = torch.argmax(
                    output,
                    dim=1
                )

                predictions.extend(
                    pred.cpu().numpy()
                )

                labels.extend(
                    graph.y.cpu().numpy()
                )

        accuracy = accuracy_score(
            labels,
            predictions
        )

        macro_f1 = f1_score(
            labels,
            predictions,
            average="macro"
        )

        weighted_f1 = f1_score(
            labels,
            predictions,
            average="weighted"
        )

        avg_loss = total_loss / len(
            train_data
        )

        scheduler.step(
            macro_f1
        )

        print(
            f"[lr={lr}] Epoch {epoch + 1}/{EPOCHS}",
            f"Loss {avg_loss:.4f}",
            f"Accuracy {accuracy:.4f}",
            f"Macro F1 {macro_f1:.4f}",
            f"Weighted F1 {weighted_f1:.4f}",
            f"LR {optimizer.param_groups[0]['lr']:.6f}"
        )

        if macro_f1 > best_f1:

            best_f1 = macro_f1
            best_weighted_f1 = weighted_f1
            best_epoch = epoch + 1

            counter = 0

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            print(
                "\nBest Model Saved (sweep checkpoint)\n"
            )

        else:

            counter += 1

            print(
                f"No improvement ({counter}/{PATIENCE})"
            )

            if counter >= PATIENCE:

                print(
                    "\nEarly stopping triggered.\n"
                )

                break

    epochs_run = epoch + 1

    sweep_results.append(
        {
            "lr": lr,
            "best_dev_macro_f1": best_f1,
            "best_dev_weighted_f1": best_weighted_f1,
            "best_epoch": best_epoch,
            "epochs_run": epochs_run,
            "checkpoint": checkpoint_path
        }
    )

    print(
        f"\nFinished run lr={lr}: "
        f"best_macro_f1={best_f1:.4f} at epoch {best_epoch}, "
        f"ran {epochs_run} epochs total.\n"
    )


print(
    "\n\n" + "=" * 78
)

print(
    "LR Sweep Results (dev set only, seed={}, epochs<={}, patience={})".format(
        SEED, EPOCHS, PATIENCE
    )
)

print(
    "=" * 78
)

header = "{:>10} | {:>16} | {:>19} | {:>11} | {:>11}".format(
    "LR", "Best Macro F1", "Best Weighted F1", "Best Epoch", "Epochs Run"
)

print(header)
print("-" * len(header))

for r in sweep_results:

    print(
        "{:>10.0e} | {:>16.4f} | {:>19.4f} | {:>11} | {:>11}".format(
            r["lr"],
            r["best_dev_macro_f1"],
            r["best_dev_weighted_f1"],
            r["best_epoch"],
            r["epochs_run"]
        )
    )

print("=" * 78)


sweep_output_path = os.path.join(
    OUTPUT_DIR,
    "lr_sweep_results.json"
)

with open(sweep_output_path, "w") as file:

    json.dump(
        sweep_results,
        file,
        indent=4
    )

print(
    "\nSaved sweep results to:",
    sweep_output_path
)

print(
    "\nNote: this sweep only evaluated on the dev set. "
    "The existing checkpoints/gat/gat_best.pt and outputs/gat/results.json "
    "from train_gat.py were not touched. No test-set evaluation was performed."
)
