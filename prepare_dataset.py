# prepare_dataset.py

import os
import shutil
import random
from utils import batch_extract_frames

# -------- Settings --------
TOTAL_PER_CLASS = 3000       # Total frames to keep per class
VAL_RATIO = 0.1
TEST_RATIO = 0.1
FRAME_RATE = 5

def clean_dirs(paths):
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f" Cleared: {path}")
        os.makedirs(path, exist_ok=True)

def extract_and_limit(video_dir, output_dir, label_prefix, frame_rate, max_frames):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Extracting from {video_dir}")

    # Temp location
    temp_dir = os.path.join("temp_extract", label_prefix)
    clean_dirs([temp_dir])

    batch_extract_frames(video_dir, temp_dir, label_prefix, frame_rate)

    # Collect all frames, shuffle and limit
    all_files = sorted([f for f in os.listdir(temp_dir) if f.endswith(('.jpg', '.png'))])
    random.shuffle(all_files)
    selected = all_files[:max_frames]

    for file in selected:
        shutil.copy(os.path.join(temp_dir, file), os.path.join(output_dir, file))

    print(f" {len(selected)} frames saved to {output_dir}")
    shutil.rmtree(temp_dir)

def split_data(source_dir, val_dir, test_dir, val_ratio, test_ratio):
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png'))])
    random.shuffle(files)

    total = len(files)
    val_count = int(total * val_ratio)
    test_count = int(total * test_ratio)

    val_split = files[:val_count]
    test_split = files[val_count:val_count + test_count]

    for f in val_split:
        shutil.copy(os.path.join(source_dir, f), os.path.join(val_dir, f))
    for f in test_split:
        shutil.copy(os.path.join(source_dir, f), os.path.join(test_dir, f))

    print(f"[SPLIT] {val_count} → val, {test_count} → test")

# -------- Configuration --------
sources = [
    {
        "video_dir": "videos/Crash-1500",
        "label_prefix": "accident",
        "train_dir": "dataset/train/Accident",
        "val_dir": "dataset/val/Accident",
        "test_dir": "dataset/test/Accident"
    },
    {
        "video_dir": "videos/Normal",
        "label_prefix": "normal",
        "train_dir": "dataset/train/Normal",
        "val_dir": "dataset/val/Normal",
        "test_dir": "dataset/test/Normal"
    }
]

# -------- Main Execution --------
if __name__ == "__main__":
    print("\n Preparing Medium & Balanced Dataset...\n")

    for src in sources:
        clean_dirs([src["train_dir"], src["val_dir"], src["test_dir"]])

        extract_and_limit(
            video_dir=src["video_dir"],
            output_dir=src["train_dir"],
            label_prefix=src["label_prefix"],
            frame_rate=FRAME_RATE,
            max_frames=TOTAL_PER_CLASS
        )

        split_data(
            source_dir=src["train_dir"],
            val_dir=src["val_dir"],
            test_dir=src["test_dir"],
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO
        )

    print("\n Dataset preparation complete with equal size and balance.\n")
