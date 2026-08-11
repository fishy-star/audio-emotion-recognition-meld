import os
import json
import pickle
import random

import joblib

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


EMBED_DIR = config.EMBEDDINGS_DIR

CHECKPOINT_DIR = os.path.join(
    config.CHECKPOINT_DIR,
    "mlp"
)

OUTPUT_DIR = os.path.join(
    config.OUTPUT_DIR,
    "mlp"
)


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



# Load embeddings

X_train = np.load(
    os.path.join(
        EMBED_DIR,
        "train_embeddings.npy"
    )
)

X_dev = np.load(
    os.path.join(
        EMBED_DIR,
        "dev_embeddings.npy"
    )
)



# Load metadata

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



# Normalize embeddings (fit on train only)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_dev = scaler.transform(X_dev)

with open(
    os.path.join(
        EMBED_DIR,
        "scaler.pkl"
    ),
    "wb"
) as file:

    pickle.dump(
        scaler,
        file
    )



# Encode labels

encoder = LabelEncoder()


y_train = encoder.fit_transform(
    train_meta["emotion"]
)


y_dev = encoder.transform(
    dev_meta["emotion"]
)


joblib.dump(
    encoder,
    os.path.join(
        EMBED_DIR,
        "label_encoder.pkl"
    )
)


print("Classes:")
print(encoder.classes_)



# Tensor conversion

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long
)

X_dev_tensor = torch.tensor(
    X_dev,
    dtype=torch.float32
)



dataset = TensorDataset(
    X_train,
    y_train_tensor
)


loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)



# Model

model = EmotionMLP(
    input_dim=768,
    num_classes=len(encoder.classes_)
)


model.to(device)



# Class weights

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)


weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(device)



criterion = FocalLoss(alpha=weights, gamma=2.0)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,
    weight_decay=1e-3
)



epochs = 10

patience = 5

counter = 0

best_f1 = 0

best_path = os.path.join(
    CHECKPOINT_DIR,
    "mlp_best.pt"
)



# Training

for epoch in range(epochs):

    model.train()

    total_loss = 0


    for X_batch, y_batch in loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)


        optimizer.zero_grad()


        output = model(
            X_batch
        )


        loss = criterion(
            output,
            y_batch
        )


        loss.backward()

        optimizer.step()


        total_loss += loss.item()



    # Validation

    model.eval()

    predictions = []

    with torch.no_grad():

        output = model(
            X_dev_tensor.to(device)
        )


        pred = torch.argmax(
            output,
            dim=1
        )


        predictions = pred.cpu().numpy()



    accuracy = accuracy_score(
        y_dev,
        predictions
    )


    macro_f1 = f1_score(
        y_dev,
        predictions,
        average="macro"
    )


    weighted_f1 = f1_score(
        y_dev,
        predictions,
        average="weighted"
    )



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


        torch.save(
            model.state_dict(),
            best_path
        )

        print("Best model saved")

    else:

        counter += 1

        print(f"No improvement ({counter}/{patience})")

        if counter >= patience:

            print("\nEarly stopping triggered.\n")

            break

epochs_run = epoch + 1


# Load best model

model.load_state_dict(
    torch.load(
        best_path,
        map_location=device
    )
)


model.eval()



with torch.no_grad():

    output = model(
        X_dev_tensor.to(device)
    )


    predictions = torch.argmax(
        output,
        dim=1
    ).cpu().numpy()



print("\nBest Model Evaluation\n")



accuracy = accuracy_score(
    y_dev,
    predictions
)


macro_f1 = f1_score(
    y_dev,
    predictions,
    average="macro"
)


weighted_f1 = f1_score(
    y_dev,
    predictions,
    average="weighted"
)



report = classification_report(
    y_dev,
    predictions,
    target_names=encoder.classes_,
    zero_division=0
)


print(report)



matrix = confusion_matrix(
    y_dev,
    predictions
)



# Save report

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
        "True Label": encoder.inverse_transform(y_dev),
        "Predicted Label": encoder.inverse_transform(predictions)
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

    "model": "mlp",

    "accuracy": accuracy,

    "macro_f1": macro_f1,

    "weighted_f1": weighted_f1,

    "best_macro_f1": best_f1,

    "num_training_samples": len(X_train),

    "num_validation_samples": len(X_dev),

    "epochs_run": epochs_run,

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


print("\nOutputs saved to:", OUTPUT_DIR)
print("✓ classification_report.txt")
print("✓ confusion_matrix.npy")
print("✓ predictions.csv")
print("✓ results.json")
print("✓ label_encoder.pkl")