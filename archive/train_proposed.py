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

from models.proposed import ProposedEmotionGAT
from utils.graph_context import build_graphs


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
    "proposed"
)


CHECKPOINT_DIR = os.path.join(
    ROOT,
    "checkpoints",
    "proposed"
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


model = ProposedEmotionGAT(

    input_dim=768,

    hidden_dim=256,

    num_classes=len(
        encoder.classes_
    )

)

model.to(device)


criterion = nn.CrossEntropyLoss(
    weight=weights,
    label_smoothing=0.1
)


optimizer = torch.optim.Adam(

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


epochs = 50

patience = 5

counter = 0

best_f1 = 0.0

best_predictions = None

best_labels = None
for epoch in range(epochs):

    random.shuffle(
        train_data
    )

    model.train()

    total_loss = 0.0


    for graph in train_data:

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
        weighted_f1
    )


    print(

        f"Epoch {epoch + 1}/{epochs}",

        f"Loss {avg_loss:.4f}",

        f"Accuracy {accuracy:.4f}",

        f"Macro F1 {macro_f1:.4f}",

        f"Weighted F1 {weighted_f1:.4f}",

        f"LR {optimizer.param_groups[0]['lr']:.6f}"

    )


    if weighted_f1 > best_f1:

        best_f1 = weighted_f1

        counter = 0

        best_predictions = predictions.copy()

        best_labels = labels.copy()


        torch.save(

            model.state_dict(),

            os.path.join(

                CHECKPOINT_DIR,

                "gat_best.pt"

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

model.load_state_dict(

    torch.load(

        os.path.join(

            CHECKPOINT_DIR,

            "gat_best.pt"

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

    "num_training_graphs": len(

        train_data

    ),

    "num_validation_graphs": len(

        dev_data

    ),

    "epochs": epochs,

    "best_weighted_f1": best_f1,

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