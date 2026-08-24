"""
Per-utterance acoustic feature extraction for MELD audio, independent of the
wav2vec2 pipeline (utils/extract_embeddings.py / extract_embeddings_multilayer.py).

Reads the same processed audio (data/processed/audio/{split}/{filename}.wav)
and the same per-split metadata CSVs already written by extract_embeddings.py
(data/embeddings/{split}_metadata.csv), so every array produced here is
row-aligned with train_embeddings.npy / train_embeddings_layers.npy -- required
for the Tier B fusion training scripts to concatenate wav2vec2 and acoustic
features by index.

Produces two aggregation formats per utterance:
  - flat:   mean + std of each frame-level feature, concatenated -> (192,)
  - frames: the frame-level feature sequence itself, unaggregated -> (T, 96)

Both are StandardScaler-normalized, fit once on train only:
  - acoustic_flat_scaler.pkl  fit on the (N, 192) flat train features
  - acoustic_frame_scaler.pkl fit on all train frames pooled together, (*, 96)
"""

import os
import json
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from spafe.features.gfcc import gfcc as spafe_gfcc
from spafe.utils.preprocessing import SlidingWindow

from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


AUDIO_DIR = os.path.join(config.PROJECT_ROOT, "data", "processed", "audio")
EMBED_DIR = config.EMBEDDINGS_DIR

SAMPLE_RATE = config.SAMPLE_RATE

N_FFT = 512
WIN_LENGTH = 400   # 25ms at 16kHz
HOP_LENGTH = 160   # 10ms at 16kHz

N_MFCC = 13
N_GFCC = 13
N_CHROMA = 12

F0_FMIN = 50.0
F0_FMAX = 500.0
F0_FRAME_LENGTH = 1024

DELTA_MAX_WIDTH = 9

GFCC_WINDOW = SlidingWindow(
    win_len=WIN_LENGTH / SAMPLE_RATE,
    win_hop=HOP_LENGTH / SAMPLE_RATE,
    win_type="hamming"
)

FEATURE_NAMES = (
    [f"mfcc_{i}" for i in range(N_MFCC)]
    + [f"gfcc_{i}" for i in range(N_GFCC)]
    + ["rms"]
    + ["f0"]
    + [f"chroma_{i}" for i in range(N_CHROMA)]
    + ["spectral_flux"]
    + ["spectral_bandwidth"]
    + ["spectral_centroid"]
    + ["zcr"]
    + [f"mfcc_delta_{i}" for i in range(N_MFCC)]
    + [f"mfcc_delta2_{i}" for i in range(N_MFCC)]
    + [f"gfcc_delta_{i}" for i in range(N_GFCC)]
    + [f"gfcc_delta2_{i}" for i in range(N_GFCC)]
)

FLAT_FEATURE_NAMES = (
    [f"{name}_mean" for name in FEATURE_NAMES]
    + [f"{name}_std" for name in FEATURE_NAMES]
)

assert len(FEATURE_NAMES) == 96
assert len(FLAT_FEATURE_NAMES) == 192


def _safe_delta(x, order):
    """librosa.feature.delta requires width <= num_frames; MELD has utterances
    short enough (down to ~7 frames at a 10ms hop) that the default width=9
    fails outright, so shrink width to fit and fall back to zeros if there
    aren't even 3 frames to differentiate across."""

    num_frames = x.shape[1]

    width = min(DELTA_MAX_WIDTH, num_frames)

    if width % 2 == 0:
        width -= 1

    if width < 3:
        return np.zeros_like(x)

    return librosa.feature.delta(x, order=order, width=width)


def _spectral_flux(magnitude):

    flux = np.zeros(magnitude.shape[1], dtype=np.float32)

    diff = np.diff(magnitude, axis=1)
    flux[1:] = np.sqrt(np.sum(diff ** 2, axis=0))

    return flux


def extract_frame_features(y, sr=SAMPLE_RATE):
    """Returns (T, 96) float32: per-frame acoustic features, hop-aligned
    across librosa's centered framing and spafe's uncentered framing by
    truncating every stream to the shortest one."""

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH
    )

    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    magnitude = np.abs(stft)

    rms = librosa.feature.rms(y=y, frame_length=WIN_LENGTH, hop_length=HOP_LENGTH)

    f0, _, _ = librosa.pyin(
        y, sr=sr, fmin=F0_FMIN, fmax=F0_FMAX,
        frame_length=F0_FRAME_LENGTH, hop_length=HOP_LENGTH, center=True
    )
    f0 = np.nan_to_num(f0, nan=0.0)[np.newaxis, :]

    chroma = librosa.feature.chroma_stft(S=magnitude, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)

    flux = _spectral_flux(magnitude)[np.newaxis, :]

    bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=WIN_LENGTH, hop_length=HOP_LENGTH)

    gfcc = spafe_gfcc(
        y.astype(np.float64), fs=sr, num_ceps=N_GFCC, nfilts=24,
        nfft=N_FFT, window=GFCC_WINDOW
    ).T

    mfcc_delta = _safe_delta(mfcc, order=1)
    mfcc_delta2 = _safe_delta(mfcc, order=2)
    gfcc_delta = _safe_delta(gfcc, order=1)
    gfcc_delta2 = _safe_delta(gfcc, order=2)

    streams = [
        mfcc, gfcc, rms, f0, chroma, flux, bandwidth, centroid, zcr,
        mfcc_delta, mfcc_delta2, gfcc_delta, gfcc_delta2
    ]

    min_t = min(stream.shape[1] for stream in streams)
    streams = [stream[:, :min_t] for stream in streams]

    frame_features = np.concatenate(streams, axis=0).T.astype(np.float32)

    return frame_features


def extract_utterance(audio_path):

    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    frame_features = extract_frame_features(y, sr)

    flat_features = np.concatenate([
        frame_features.mean(axis=0),
        frame_features.std(axis=0)
    ]).astype(np.float32)

    return flat_features, frame_features


def extract_split(split, num_workers):

    meta = pd.read_csv(os.path.join(EMBED_DIR, f"{split}_metadata.csv"))

    audio_paths = [
        os.path.join(AUDIO_DIR, split, filename)
        for filename in meta["filename"]
    ]

    missing = [p for p in audio_paths if not os.path.exists(p)]
    assert not missing, f"Missing audio files for {split}: {missing[:5]} ..."

    print(f"\nExtracting acoustic features for {split}: {len(audio_paths)} files")

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(extract_utterance, audio_paths, chunksize=8),
            total=len(audio_paths)
        ))

    flat_array = np.stack([r[0] for r in results]).astype(np.float32)

    frame_array = np.empty(len(results), dtype=object)
    for i, r in enumerate(results):
        frame_array[i] = r[1]

    return flat_array, frame_array


def main():

    num_workers = max(1, (os.cpu_count() or 2) - 1)
    print("Using", num_workers, "worker processes")

    train_flat, train_frames = extract_split("train", num_workers)
    dev_flat, dev_frames = extract_split("dev", num_workers)
    test_flat, test_frames = extract_split("test", num_workers)

    # Flat-aggregate scaler: fit on train only, transform dev/test
    flat_scaler = StandardScaler()

    train_flat = flat_scaler.fit_transform(train_flat).astype(np.float32)
    dev_flat = flat_scaler.transform(dev_flat).astype(np.float32)
    test_flat = flat_scaler.transform(test_flat).astype(np.float32)

    with open(os.path.join(EMBED_DIR, "acoustic_flat_scaler.pkl"), "wb") as f:
        pickle.dump(flat_scaler, f)

    # Frame-level scaler: fit on all train frames pooled together, transform
    # each utterance's frames (train/dev/test) with that same fit.
    frame_scaler = StandardScaler()
    frame_scaler.fit(np.concatenate(list(train_frames), axis=0))

    with open(os.path.join(EMBED_DIR, "acoustic_frame_scaler.pkl"), "wb") as f:
        pickle.dump(frame_scaler, f)

    def scale_frames(frame_array):

        scaled = np.empty(len(frame_array), dtype=object)

        for i, frames in enumerate(frame_array):
            scaled[i] = frame_scaler.transform(frames).astype(np.float32)

        return scaled

    train_frames = scale_frames(train_frames)
    dev_frames = scale_frames(dev_frames)
    test_frames = scale_frames(test_frames)

    np.save(os.path.join(EMBED_DIR, "train_acoustic_flat.npy"), train_flat)
    np.save(os.path.join(EMBED_DIR, "dev_acoustic_flat.npy"), dev_flat)
    np.save(os.path.join(EMBED_DIR, "test_acoustic_flat.npy"), test_flat)

    np.save(os.path.join(EMBED_DIR, "train_acoustic_frames.npy"), train_frames, allow_pickle=True)
    np.save(os.path.join(EMBED_DIR, "dev_acoustic_frames.npy"), dev_frames, allow_pickle=True)
    np.save(os.path.join(EMBED_DIR, "test_acoustic_frames.npy"), test_frames, allow_pickle=True)

    with open(os.path.join(EMBED_DIR, "acoustic_feature_names.json"), "w") as f:
        json.dump(
            {"frame_features": FEATURE_NAMES, "flat_features": FLAT_FEATURE_NAMES},
            f,
            indent=2
        )

    print("\nAcoustic feature extraction complete.")
    print("train_acoustic_flat:", train_flat.shape)
    print("dev_acoustic_flat:", dev_flat.shape)
    print("test_acoustic_flat:", test_flat.shape)
    print("train_acoustic_frames:", len(train_frames), "utterances, feature dim", train_frames[0].shape[1])


if __name__ == "__main__":
    main()
