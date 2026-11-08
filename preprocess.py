"""Resize raw images to 224x224 and skip unreadable files.

Usage:
    python preprocess.py --src dataset_raw --dst dataset_preprocessed

Expects dataset_raw/<class>/*.jpg for each class in config.CLASSES.
"""
import argparse
import os

import cv2

from config import CLASSES, IMG_SIZE


def preprocess_images(src, dst):
    for category in CLASSES:
        in_dir = os.path.join(src, category)
        out_dir = os.path.join(dst, category)
        os.makedirs(out_dir, exist_ok=True)

        saved = skipped = 0
        for name in os.listdir(in_dir):
            img = cv2.imread(os.path.join(in_dir, name))
            if img is None:
                skipped += 1
                continue
            cv2.imwrite(os.path.join(out_dir, name), cv2.resize(img, IMG_SIZE))
            saved += 1
        print(f"{category}: {saved} saved, {skipped} unreadable skipped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize and clean the raw dataset.")
    parser.add_argument("--src", default="dataset_raw")
    parser.add_argument("--dst", default="dataset_preprocessed")
    args = parser.parse_args()
    preprocess_images(args.src, args.dst)
