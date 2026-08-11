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

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from models.bilstm import EmotionBiLSTM
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


class EmotionBiLSTMWithLayerWeighting(nn.Module):
    """Swaps the fixed last-layer-mean-pool embedding for a trainable
    softmax-weighted sum across all 13 wav2vec2 layers, applied per
    utterance before the unmodified EmotionBiLSTM."""

    def __init__(self, input_dim=768, hidden_dim=256, num_classes=7, num_layers=13):
        super().__init__()

        self.layer_weighting = TrainableLayerWeighting(num_layers=num_layers)
        self.bilstm = EmotionBiLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )

    def forward(self, x, lengths):
        # x: (batch, seq_len, num_layers, hidden_dim)
        pooled = self.layer_weighting(x)
        return self.bilstm(pooled, lengths)


EMBED_DIR = config.EMBEDDINGS_DIR


OUTPUT_DIR = os.path.join(
    config.OUTPUT_DIR,
    "bilstm_v2"
)


CHECKPOINT_DIR = os.path.join(
    config.CHECKPOINT_DIR,
    "bilstm_v2"
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


print("Using:", device)



# Seed

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)



class DialogueDataset(Dataset):

    def __init__(
        self,
        embeddings,
        metadata,
        encoder
    ):

        self.dialogues = []

        metadata = metadata.copy()


        metadata["label"] = encoder.transform(
            metadata["emotion"]
        )


        for _, dialogue in metadata.groupby(
            "dialogue_id",
            sort=False
        ):


            dialogue = dialogue.sort_values(
                "utterance_id"
            )


            indices = dialogue.index.tolist()


            x = embeddings[
                indices
            ]


            y = dialogue["label"].values



            self.dialogues.append(
                (
                    torch.tensor(
                        x,
                        dtype=torch.float32
                    ),

                    torch.tensor(
                        y,
                        dtype=torch.long
                    )
                )
            )



    def __len__(self):

        return len(self.dialogues)



    def __getitem__(
        self,
        idx
    ):

        return self.dialogues[idx]





def collate_fn(batch):

    xs = []
    ys = []


    for x, y in batch:

        xs.append(x)
        ys.append(y)



    lengths = torch.tensor(
        [
            len(x)
            for x in xs
        ]
    )


    xs = pad_sequence(
        xs,
        batch_first=True
    )


    ys = pad_sequence(
        ys,
        batch_first=True,
        padding_value=-100
    )


    return xs, ys, lengths





# Load ALL-LAYER embeddings (13, 768) per utterance, not the single
# last-layer mean-pooled embeddings used by train_bilstm.py

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




# Label encoder (shared, fit on train only by train_mlp.py) -- read only,
# never written back to data/embeddings/label_encoder.pkl.

encoder = joblib.load(
    os.path.join(
        EMBED_DIR,
        "label_encoder.pkl"
    )
)



print(
    "Classes:",
    encoder.classes_
)




train_dataset = DialogueDataset(
    train_embeddings,
    train_meta,
    encoder
)


dev_dataset = DialogueDataset(
    dev_embeddings,
    dev_meta,
    encoder
)




train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn
)


dev_loader = DataLoader(
    dev_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_fn
)




# Model

model = EmotionBiLSTMWithLayerWeighting(
    input_dim=hidden_dim,
    hidden_dim=256,
    num_classes=len(
        encoder.classes_
    ),
    num_layers=num_layers
)


model.to(device)





# Class weights

labels = encoder.transform(
    train_meta["emotion"]
)


weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)


weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(device)




criterion = FocalLoss(
    alpha=weights,
    gamma=2.0,
    ignore_index=-100
)



optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,
    weight_decay=1e-3
)





epochs = 40

patience = 8

counter = 0

best_f1 = 0


best_path = os.path.join(
    CHECKPOINT_DIR,
    "bilstm_v2_best.pt"
)





# Training

for epoch in range(epochs):


    model.train()


    total_loss = 0



    for x, y, lengths in train_loader:


        x = x.to(device)
        y = y.to(device)


        optimizer.zero_grad()



        output = model(
            x,
            lengths
        )



        loss = criterion(
            output.reshape(-1, len(encoder.classes_)),
            y.reshape(-1)
        )



        loss.backward()



        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )



        optimizer.step()



        total_loss += loss.item()





    # Validation

    model.eval()


    predictions = []
    ground_truth = []



    with torch.no_grad():


        for x, y, lengths in dev_loader:


            x = x.to(device)



            output = model(
                x,
                lengths
            )



            pred = torch.argmax(
                output,
                dim=-1
            ).cpu()



            mask = y != -100



            predictions.extend(
                pred[mask].numpy()
            )


            ground_truth.extend(
                y[mask].numpy()
            )





    accuracy = accuracy_score(
        ground_truth,
        predictions
    )


    macro_f1 = f1_score(
        ground_truth,
        predictions,
        average="macro"
    )


    weighted_f1 = f1_score(
        ground_truth,
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



predictions = []
ground_truth = []



with torch.no_grad():


    for x, y, lengths in dev_loader:


        x = x.to(device)


        output = model(
            x,
            lengths
        )


        pred = torch.argmax(
            output,
            dim=-1
        ).cpu()



        mask = y != -100



        predictions.extend(
            pred[mask].numpy()
        )


        ground_truth.extend(
            y[mask].numpy()
        )





print("\nBest Model Evaluation\n")



report = classification_report(
    ground_truth,
    predictions,
    target_names=encoder.classes_,
    zero_division=0
)



print(report)




matrix = confusion_matrix(
    ground_truth,
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
        "True Label": encoder.inverse_transform(
            ground_truth
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


learned_layer_weights = model.layer_weighting.get_weights().detach().cpu().tolist()



results = {

    "model": "bilstm_v2_trainable_layer_weighting",

    "embedding_strategy": "trainable_softmax_weighted_sum_across_13_layers",

    "accuracy": accuracy_score(
        ground_truth,
        predictions
    ),

    "macro_f1": f1_score(
        ground_truth,
        predictions,
        average="macro"
    ),

    "weighted_f1": f1_score(
        ground_truth,
        predictions,
        average="weighted"
    ),

    "best_macro_f1": best_f1,

    "num_training_samples": len(
        train_dataset
    ),

    "num_validation_samples": len(
        dev_dataset
    ),

    "epochs_run": epochs_run,

    "patience": patience,

    "learned_layer_weights": learned_layer_weights,

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
print("✓ scaler_v2.pkl")
