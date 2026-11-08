"""Split preprocessed images into train/ and test/ folders (80/20 per class).

Usage:
    python split_dataset.py --src dataset_preprocessed --dst dataset_split
"""
import argparse
import os
import shutil

from sklearn.model_selection import train_test_split

from config import CLASSES


def split_dataset(src, dst, test_size=0.2):
    for category in CLASSES:
        in_dir = os.path.join(src, category)
        train_dir = os.path.join(dst, "train", category)
        test_dir = os.path.join(dst, "test", category)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        images = sorted(os.listdir(in_dir))
        train_imgs, test_imgs = train_test_split(images, test_size=test_size, random_state=42)
        for name in train_imgs:
            shutil.copy(os.path.join(in_dir, name), os.path.join(train_dir, name))
        for name in test_imgs:
            shutil.copy(os.path.join(in_dir, name), os.path.join(test_dir, name))
        print(f"{category}: {len(train_imgs)} train, {len(test_imgs)} test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a reproducible train/test split.")
    parser.add_argument("--src", default="dataset_preprocessed")
    parser.add_argument("--dst", default="dataset_split")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    split_dataset(args.src, args.dst, args.test_size)
