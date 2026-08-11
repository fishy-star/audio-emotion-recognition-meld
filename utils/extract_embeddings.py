import os
import torch
import torchaudio
import pandas as pd
import numpy as np

from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

AUDIO_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "audio"
)

RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "MELD.Raw"
)

EMBED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "embeddings"
)

os.makedirs(EMBED_DIR, exist_ok=True)


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using:", device)


processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

model.to(device)
model.eval()


SPLITS = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv"
}


def extract_embeddings(split):

    print(f"\nProcessing {split}")

    csv_path = os.path.join(
        RAW_DIR,
        SPLITS[split]
    )

    df = pd.read_csv(csv_path)

    embeddings = []
    metadata = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        dialogue_id = row["Dialogue_ID"]
        utterance_id = row["Utterance_ID"]

        filename = f"dia{dialogue_id}_utt{utterance_id}.wav"

        audio_path = os.path.join(
            AUDIO_DIR,
            split,
            filename
        )

        if not os.path.exists(audio_path):
            print("Missing:", audio_path)
            continue

        waveform, sample_rate = torchaudio.load(audio_path)

        inputs = processor(
            waveform.squeeze(),
            sampling_rate=sample_rate,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.last_hidden_state

        embedding = hidden_states.mean(dim=1)

        embedding = embedding.cpu().numpy().squeeze()

        embeddings.append(embedding)

        metadata.append({
            "filename": filename,
            "dialogue_id": dialogue_id,
            "utterance_id": utterance_id,
            "emotion": row["Emotion"]
        })

    embeddings = np.array(embeddings)
    metadata = pd.DataFrame(metadata)

    np.save(
        os.path.join(
            EMBED_DIR,
            f"{split}_embeddings.npy"
        ),
        embeddings
    )

    metadata.to_csv(
        os.path.join(
            EMBED_DIR,
            f"{split}_metadata.csv"
        ),
        index=False
    )

    print(f"{split} embeddings:", embeddings.shape)


if __name__ == "__main__":

    for split in SPLITS:
        extract_embeddings(split)

    print("Embedding extraction complete!")