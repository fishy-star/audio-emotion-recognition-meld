import os

import pandas as pd


ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT, "data", "raw", "MELD.Raw")
EMBED_DIR = os.path.join(ROOT, "data", "embeddings")


# MELD original metadata
train_meld = pd.read_csv(
    os.path.join(RAW_DIR, "train_sent_emo.csv")
)

dev_meld = pd.read_csv(
    os.path.join(RAW_DIR, "dev_sent_emo.csv")
)


# Embedding metadata to merge the speaker column into
train_meta = pd.read_csv(
    os.path.join(EMBED_DIR, "train_metadata.csv")
)

dev_meta = pd.read_csv(
    os.path.join(EMBED_DIR, "dev_metadata.csv")
)


# Prepare speaker information

train_speaker = train_meld[
    [
        "Dialogue_ID",
        "Utterance_ID",
        "Speaker"
    ]
].rename(
    columns={
        "Dialogue_ID": "dialogue_id",
        "Utterance_ID": "utterance_id",
        "Speaker": "speaker"
    }
)


dev_speaker = dev_meld[
    [
        "Dialogue_ID",
        "Utterance_ID",
        "Speaker"
    ]
].rename(
    columns={
        "Dialogue_ID": "dialogue_id",
        "Utterance_ID": "utterance_id",
        "Speaker": "speaker"
    }
)


# Merge speaker into metadata

train_meta = train_meta.merge(
    train_speaker,
    on=[
        "dialogue_id",
        "utterance_id"
    ],
    how="left"
)


dev_meta = dev_meta.merge(
    dev_speaker,
    on=[
        "dialogue_id",
        "utterance_id"
    ],
    how="left"
)


print(train_meta.head())
print(dev_meta.head())


print(
    "Missing train speakers:",
    train_meta["speaker"].isna().sum()
)

print(
    "Missing dev speakers:",
    dev_meta["speaker"].isna().sum()
)


# Save

train_meta.to_csv(
    os.path.join(EMBED_DIR, "train_metadata.csv"),
    index=False
)

dev_meta.to_csv(
    os.path.join(EMBED_DIR, "dev_metadata.csv"),
    index=False
)


print("Metadata fixed!")

