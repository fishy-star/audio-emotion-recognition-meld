import os
import json
import random
import pickle

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

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
from torch_geometric.loader import DataLoader

from models.proposed_typed import ProposedTypedEmotionGAT
from utils.graph_typed import build_graphs


ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

EMBED_DIR = os.path.join(
    ROOT,
    "data",
    "embeddings"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "outputs",
    "proposed_typed"
)

CHECKPOINT_DIR = os.path.join(
    ROOT,
    "checkpoints",
    "proposed_typed"
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


seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


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

print(train_meta.columns)


train_graphs, encoder = build_graphs(
    train_embeddings,
    train_meta,
    encoder=None
)

dev_graphs, _ = build_graphs(
    dev_embeddings,
    dev_meta,
    encoder=encoder
)


for graph in train_graphs[:3]:

    print(
        "Edge types:",
        np.unique(
            graph["edge_type"],
            return_counts=True
        )
    )


def convert_to_data(graphs):

    data_list = []

    for graph in graphs:

        data_list.append(

            Data(

                x=torch.tensor(
                    graph["x"],
                    dtype=torch.float32
                ),

                edge_index=torch.tensor(
                    graph["edge_index"],
                    dtype=torch.long
                ),

                edge_type=torch.tensor(
                    graph["edge_type"],
                    dtype=torch.long
                ),

                speaker_ids=torch.tensor(
                    graph["speaker_ids"],
                    dtype=torch.long
                ),

                y=torch.tensor(
                    graph["y"],
                    dtype=torch.long
                )

            )

        )

    return data_list


train_data = convert_to_data(
    train_graphs
)

dev_data = convert_to_data(
    dev_graphs
)

print(
    "Training Graphs:",
    len(train_data)
)

print(
    "Validation Graphs:",
    len(dev_data)
)


train_loader = DataLoader(
    train_data,
    batch_size=8,
    shuffle=True
)

dev_loader = DataLoader(
    dev_data,
    batch_size=8,
    shuffle=False
)


train_labels = np.concatenate(
    [
        graph["y"]
        for graph in train_graphs
    ]
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(device)

print(
    "Classes:",
    encoder.classes_
)

print(
    "Weights:",
    weights
)
# Model

model = ProposedTypedEmotionGAT(

    input_dim=768,

    hidden_dim=256,

    num_classes=len(
        encoder.classes_
    ),

    heads=4,

    num_relations=4,

    dropout=0.2

).to(device)


criterion = nn.CrossEntropyLoss(
    weight=weights
)


optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=5e-4,

    weight_decay=1e-3

)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=3

)


epochs = 50

patience = 5

counter = 0

best_f1 = 0.0


best_path = os.path.join(

    CHECKPOINT_DIR,

    "proposed_typed_best.pt"

)


for epoch in range(epochs):

    model.train()

    total_loss = 0.0


    for batch in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        output = model(

            batch.x,

            batch.edge_index,

            batch.edge_type,

            batch.speaker_ids

        )

        loss = criterion(

            output,

            batch.y.view(-1)

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

        for batch in dev_loader:

            batch = batch.to(device)

            output = model(

                batch.x,

                batch.edge_index,

                batch.edge_type,

                batch.speaker_ids

            )

            pred = torch.argmax(

                output,

                dim=1

            )

            predictions.extend(

                pred.cpu().numpy()

            )

            labels.extend(

                batch.y.view(-1)

                .cpu()

                .numpy()

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


    scheduler.step(

        weighted_f1

    )


    print(

        f"Epoch {epoch+1}/{epochs} | "

        f"Loss: {total_loss/len(train_loader):.4f} | "

        f"Accuracy: {accuracy:.4f} | "

        f"Macro F1: {macro_f1:.4f} | "

        f"Weighted F1: {weighted_f1:.4f} | "

        f"LR: {optimizer.param_groups[0]['lr']:.6f}"

    )


    if weighted_f1 > best_f1:

        best_f1 = weighted_f1

        counter = 0

        torch.save(

            model.state_dict(),

            best_path

        )

        print(

            "Best model saved"

        )

    else:

        counter += 1

        print(

            f"No improvement ({counter}/{patience})"

        )

        if counter >= patience:

            print(

                "Early stopping triggered"

            )

            break
# Load best model

model.load_state_dict(

    torch.load(

        best_path,

        map_location=device

    )

)


model.eval()


predictions = []

labels = []


with torch.no_grad():

    for batch in dev_loader:

        batch = batch.to(device)

        output = model(

            batch.x,

            batch.edge_index,

            batch.edge_type,

            batch.speaker_ids

        )

        pred = torch.argmax(

            output,

            dim=1

        )

        predictions.extend(

            pred.cpu().numpy()

        )

        labels.extend(

            batch.y.view(-1)

            .cpu()

            .numpy()

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

print(report)


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

    file.write(report)


np.save(

    os.path.join(

        OUTPUT_DIR,

        "confusion_matrix.npy"

    ),

    matrix

)


prediction_df = pd.DataFrame(

    {

        "True Label":

            encoder.inverse_transform(

                labels

            ),

        "Predicted Label":

            encoder.inverse_transform(

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

    "accuracy":

        accuracy_score(

            labels,

            predictions

        ),

    "macro_f1":

        f1_score(

            labels,

            predictions,

            average="macro"

        ),

    "weighted_f1":

        f1_score(

            labels,

            predictions,

            average="weighted"

        ),

    "best_weighted_f1":

        best_f1,

    "training_graphs":

        len(train_data),

    "validation_graphs":

        len(dev_data),

    "classes":

        list(

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

print("✓ classification_report.txt")
print("✓ confusion_matrix.npy")
print("✓ predictions.csv")
print("✓ results.json")
print("✓ label_encoder.pkl")