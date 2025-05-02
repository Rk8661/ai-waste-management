import os
import cv2
import numpy as np

# Define dataset path
DATASET_PATH = "/Users/rohankumbhar/dataset2"
OUTPUT_PATH = "/Users/rohankumbhar/trained"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Image parameters
IMAGE_SIZE = (224, 224)  # Resize all images to 224x224

# Categories (Folder Names)
CATEGORIES = ["metal", "plastic", "paper"]

# Function to preprocess imagesp
def preprocess_images():
    for category in CATEGORIES:
        input_folder = os.path.join(DATASET_PATH, category)
        output_folder = os.path.join(OUTPUT_PATH, category)

        os.makedirs(output_folder, exist_ok=True)

        for img_name in os.listdir(input_folder):
            img_path = os.path.join(input_folder, img_name)

            # Read image
            img = cv2.imread(img_path)

            if img is None:
                print(f"Skipping {img_name}, unable to read image.")
                continue

            # Resize image
            img = cv2.resize(img, IMAGE_SIZE)

            # Normalize pixel values (0 to 1)
            img = img / 255.0

            # Save preprocessed image
            save_path = os.path.join(output_folder, img_name)
            cv2.imwrite(save_path, img * 255)  # Convert back to 0-255 range before saving

            print(f"Processed and saved: {save_path}")

# Run preprocessing
if __name__ == "__main__":
    preprocess_images()
    print("All images preprocessed successfully!")