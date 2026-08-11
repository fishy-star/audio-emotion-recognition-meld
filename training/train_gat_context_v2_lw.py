import os
import json
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

from sklearn.preprocessing import StandardScaler

from models.gat import EmotionGAT
from models.gat_v2 import drop_edges
from utils.graph_context import build_graphs
from utils.embedding_strategies import TrainableLayerWeighting


SEED = int(os.environ.get("SEED", 42))

torch.manual_seed(SEED)
np.random.seed(SEED)


class EmotionGATWithLayerWeighting(nn.Module):
    """Swaps the fixed last-layer-mean-pool node features for a trainable
    softmax-weighted sum across all 13 wav2vec2 layers, applied per node
    before the unmodified EmotionGAT."""

    def __init__(self, input_dim=768, num_classes=7, num_layers=13):
        super().__init__()

        self.layer_weighting = TrainableLayerWeighting(num_layers=num_layers)
        self.gat = EmotionGAT(input_dim=input_dim, num_classes=num_classes)

    def forward(self, x, edge_index):
        # x: (num_nodes, num_layers, hidden_dim)
        pooled = self.layer_weighting(x)
        return self.gat(pooled, edge_index)


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


# DropEdge rate, unchanged from train_gat_context_v2.py -- this run tests
# DropEdge combined with trainable wav2vec2 layer fusion, the isolated
# combination that no surviving artifact in this repo actually verifies.
DROP_EDGE_P = 0.15


EMBED_DIR = config.EMBEDDINGS_DIR


OUTPUT_DIR = os.path.join(
    config.OUTPUT_DIR,
    "gat_context_v2_lw"
)


CHECKPOINT_DIR = os.path.join(
    config.CHECKPOINT_DIR,
    "gat_context_v2_lw"
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


# Load ALL-LAYER embeddings (13, 768) per utterance, not the single
# last-layer mean-pooled embeddings used by train_gat_context_v2.py

train_embeddings = np.load(
    os.path.join(
        EMBED_DIR,
        "train_embeddings_layers.npy"
    )
)


dev_embeddings = np.load(
    os.path.join(
        EMBED_DIR,
        "dev_embeddings_layers.npy"
    )
)

num_layers = train_embeddings.shape[1]
hidden_dim = train_embeddings.shape[2]

print("Train layer embeddings:", train_embeddings.shape)
print("Dev layer embeddings:", dev_embeddings.shape)


# Normalize embeddings (fit on train only). Fit a NEW scaler over the
# flattened (num_layers * hidden_dim) features -- do NOT reuse or
# overwrite data/embeddings/scaler.pkl, which was fit on the single
# last-layer mean-pooled embeddings and is shared with other scripts.

scaler = StandardScaler()

train_flat = train_embeddings.reshape(train_embeddings.shape[0], -1)
dev_flat = dev_embeddings.reshape(dev_embeddings.shape[0], -1)

train_flat = scaler.fit_transform(train_flat)
dev_flat = scaler.transform(dev_flat)

train_embeddings = train_flat.reshape(-1, num_layers, hidden_dim)
dev_embeddings = dev_flat.reshape(-1, num_layers, hidden_dim)

with open(
    os.path.join(
        OUTPUT_DIR,
        "scaler_v2.pkl"
    ),
    "wb"
) as file:

    pickle.dump(
        scaler,
        file
    )


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


model = EmotionGATWithLayerWeighting(

    input_dim=hidden_dim,

    num_classes=len(
        encoder.classes_
    ),

    num_layers=num_layers

)

model.to(device)


criterion = FocalLoss(
    alpha=weights,
    gamma=2.0
)


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

best_predictions = None

best_labels = None
for epoch in range(epochs):

    model.train()

    total_loss = 0.0


    for idx in train_sampler:

        graph = train_data[idx]

        graph = graph.to(device)


        optimizer.zero_grad()


        # DropEdge: randomly drop a fraction of non-self-loop edges each step,
        # regularizing against overfitting to specific dialogue structures.
        edge_index = drop_edges(
            graph.edge_index,
            p=DROP_EDGE_P,
            training=True
        )


        output = model(

            graph.x,

            edge_index

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


            # full graph at eval time — DropEdge is a training-only regularizer
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

        best_predictions = predictions.copy()

        best_labels = labels.copy()


        torch.save(

            model.state_dict(),

            os.path.join(

                CHECKPOINT_DIR,

                "gat_context_v2_lw_best.pt"

            )

        )


        print(

            "\nBest Model Saved\n"

        )


    else:

        counter += 1


        print(

            f"No improvement ({counter}/{patience})"

        )


        if counter >= patience:

            print(

                "\nEarly stopping triggered.\n"

            )

            break

epochs_run = epoch + 1

model.load_state_dict(

    torch.load(

        os.path.join(

            CHECKPOINT_DIR,

            "gat_context_v2_lw_best.pt"

        ),

        map_location=device

    )

)


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


print(

    "\nBest Model Evaluation\n"

)


report = classification_report(

    labels,

    predictions,

    target_names=encoder.classes_,

    zero_division=0

)


print(

    report

)


matrix = confusion_matrix(

    labels,

    predictions

)


with open(

    os.path.join(

        OUTPUT_DIR,

        "classification_report.txt"

    ),

    "w"

) as file:

    file.write(

        report

    )


np.save(

    os.path.join(

        OUTPUT_DIR,

        "confusion_matrix.npy"

    ),

    matrix

)


prediction_df = pd.DataFrame(

    {

        "True Label": encoder.inverse_transform(

            labels

        ),

        "Predicted Label": encoder.inverse_transform(

            predictions

        )

    }

)


prediction_df.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "predictions.csv"

    ),

    index=False

)


with open(

    os.path.join(

        OUTPUT_DIR,

        "label_encoder.pkl"

    ),

    "wb"

) as file:

    pickle.dump(

        encoder,

        file

    )


results = {

    "model": "gat_context_v2_lw_trainable_layer_weighting",

    "embedding_strategy": "trainable_softmax_weighted_sum_across_13_layers",

    "accuracy": accuracy_score(

        labels,

        predictions

    ),

    "macro_f1": f1_score(

        labels,

        predictions,

        average="macro"

    ),

    "weighted_f1": f1_score(

        labels,

        predictions,

        average="weighted"

    ),

    "num_training_samples": len(

        train_data

    ),

    "num_validation_samples": len(

        dev_data

    ),

    "epochs_run": epochs_run,

    "best_macro_f1": best_f1,

    "drop_edge_p": DROP_EDGE_P,

    "seed": SEED,

    "patience": patience,

    "learned_layer_weights": model.layer_weighting.get_weights().detach().cpu().tolist(),

    "classes": list(

        encoder.classes_

    )

}


with open(

    os.path.join(

        OUTPUT_DIR,

        "results.json"

    ),

    "w"

) as file:

    json.dump(

        results,

        file,

        indent=4

    )


print(

    "\nOutputs saved to:",

    OUTPUT_DIR

)

print(

    "✓ classification_report.txt"

)

print(

    "✓ confusion_matrix.npy"

)

print(

    "✓ predictions.csv"

)

print(

    "✓ results.json"

)

print(

    "✓ label_encoder.pkl"

)
