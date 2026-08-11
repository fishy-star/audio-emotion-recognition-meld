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

AUDIO_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "audio")
EMBED_DIR = os.path.join(PROJECT_ROOT, "data", "embeddings")

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using:", device)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.to(device)
model.eval()


def extract_layers(split):

    print(f"\nProcessing {split} (all-layer)")

    metadata = pd.read_csv(
        os.path.join(EMBED_DIR, f"{split}_metadata.csv")
    )

    all_layer_embeddings = []

    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):

        audio_path = os.path.join(AUDIO_DIR, split, row["filename"])

        waveform, sample_rate = torchaudio.load(audio_path)

        inputs = processor(
            waveform.squeeze(),
            sampling_rate=sample_rate,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # 13 layers x (1, T, 768) -> mean-pool over time -> (13, 768)
        layers = torch.stack(
            [h.mean(dim=1).squeeze(0) for h in outputs.hidden_states],
            dim=0
        )

        all_layer_embeddings.append(layers.cpu().numpy())

    all_layer_embeddings = np.array(all_layer_embeddings, dtype=np.float32)

    np.save(
        os.path.join(EMBED_DIR, f"{split}_embeddings_layers.npy"),
        all_layer_embeddings
    )

    print(f"{split} layer embeddings:", all_layer_embeddings.shape)


if __name__ == "__main__":

    for split in ["train", "dev"]:
        extract_layers(split)

    print("Multi-layer embedding extraction complete!")
