import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

# CHANGE THIS PATH to the one printed by kagglehub
CSV_PATH = "/Users/ibrahim/.cache/kagglehub/datasets/genadieva/fer-2013-csv-dataset/versions/1/fer2013.csv"

OUTPUT_DIR = "data/fer2013"

EMOTION_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

def create_dirs():
    for split in ["train", "test"]:
        for emotion in EMOTION_MAP.values():
            os.makedirs(os.path.join(OUTPUT_DIR, split, emotion), exist_ok=True)

def main():
    create_dirs()
    df = pd.read_csv(CSV_PATH)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        emotion = EMOTION_MAP[row["emotion"]]
        pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)

        if row["Usage"] == "Training":
            split = "train"
        else:
            split = "test"

        img_path = os.path.join(
            OUTPUT_DIR, split, emotion, f"{idx}.jpg"
        )

        cv2.imwrite(img_path, pixels)

    print("✅ FER-2013 conversion completed successfully.")

if __name__ == "__main__":
    main()