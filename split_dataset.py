import os
import numpy as np
import cv2
import shutil
from sklearn.model_selection import train_test_split

# Paths
PREPROCESSED_DATASET_PATH = "/Users/rohankumbhar/trained"
TRAIN_PATH = "/Users/rohankumbhar/dataset_split/train"
TEST_PATH = "/Users/rohankumbhar/dataset_split/test"

# Create train and test directories
os.makedirs(TRAIN_PATH, exist_ok=True)
os.makedirs(TEST_PATH, exist_ok=True)


# Categories (Plastic, Metal, Paper)
CATEGORIES = ["metal", "plastic", "paper"]

# Function to split and save images
def split_dataset(test_size=0.2):
    for category in CATEGORIES:
        input_folder = os.path.join(PREPROCESSED_DATASET_PATH, category)
        train_folder = os.path.join(TRAIN_PATH, category)
        test_folder = os.path.join(TEST_PATH, category)

        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        # Get list of all images
        images = os.listdir(input_folder)

        # Split into training (80%) and testing (20%) sets
        train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)

        # Move images to respective folders
        for img_name in train_images:
            shutil.copy(os.path.join(input_folder, img_name), os.path.join(train_folder, img_name))

        for img_name in test_images:
            shutil.copy(os.path.join(input_folder, img_name), os.path.join(test_folder, img_name))

        print(f"Category '{category}': {len(train_images)} training images, {len(test_images)} testing images.")

# Run the split function
if __name__ == "__main__":
    split_dataset()
    print("Dataset successfully split into training and testing sets!")