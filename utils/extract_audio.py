import os
import pandas as pd
import subprocess
from tqdm import tqdm


# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "MELD.Raw")

SPLITS = {
    "train": {
        "csv": "train_sent_emo.csv",
        "video_dir": "train_splits",
        "output_dir": os.path.join(PROJECT_ROOT, "data", "processed", "audio", "train")
    },
    "dev": {
        "csv": "dev_sent_emo.csv",
        "video_dir": "dev_splits_complete",
        "output_dir": os.path.join(PROJECT_ROOT, "data", "processed", "audio", "dev")
    },
    "test": {
        "csv": "test_sent_emo.csv",
        "video_dir": "output_repeated_splits_test",
        "output_dir": os.path.join(PROJECT_ROOT, "data", "processed", "audio", "test")
    }
}


def extract_audio(split_name, config):

    csv_path = os.path.join(RAW_DIR, config["csv"])
    video_dir = os.path.join(RAW_DIR, config["video_dir"])
    output_dir = config["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    print(f"\nProcessing {split_name}: {len(df)} files")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        dialogue_id = row["Dialogue_ID"]
        utterance_id = row["Utterance_ID"]

        filename = f"dia{dialogue_id}_utt{utterance_id}"

        video_path = os.path.join(
            video_dir,
            filename + ".mp4"
        )

        audio_path = os.path.join(
            output_dir,
            filename + ".wav"
        )

        # skip already processed files
        if os.path.exists(audio_path):
            continue

        if not os.path.exists(video_path):
            print(f"Missing: {video_path}")
            continue

        command = [
            "ffmpeg",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-y",
            audio_path
        ]

        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )


if __name__ == "__main__":

    for split, config in SPLITS.items():
        extract_audio(split, config)

    print("\nAudio extraction complete!")