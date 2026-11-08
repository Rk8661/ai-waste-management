"""Fine-tune MobileNetV2 to classify waste as metal, plastic or paper.

Usage:
    python train_model.py --data dataset_split [--epochs 25]

Expects dataset_split/train/<class>/*.jpg and dataset_split/test/<class>/*.jpg
(see split_dataset.py). 15% of the training images are held out for
validation and early stopping, so the test set is only touched once, at the
end, and the reported test accuracy is an honest estimate.
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import CLASSES, IMG_SIZE
from model import build_model

BATCH_SIZE = 32
VAL_SPLIT = 0.15
SEED = 42


def build_generators(data_dir):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    augment = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
        validation_split=VAL_SPLIT,
    )
    # Same split fraction, no augmentation: validation images are the same
    # files Keras excludes from the training subset above.
    plain_split = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=VAL_SPLIT)
    plain = ImageDataGenerator(rescale=1.0 / 255.0)

    common = dict(target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", classes=CLASSES)
    train_gen = augment.flow_from_directory(train_dir, subset="training", seed=SEED, **common)
    val_gen = plain_split.flow_from_directory(train_dir, subset="validation", shuffle=False, **common)
    # shuffle=False keeps predictions aligned with test_gen.classes for the confusion matrix
    test_gen = plain.flow_from_directory(test_dir, shuffle=False, **common)
    print("Class indices:", train_gen.class_indices)
    return train_gen, val_gen, test_gen


def saliency_box(model, image_path, out_path):
    """Rough object localisation: box around the pixels with the largest gradient."""
    import cv2

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    arr = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img), 0) / 255.0
    tensor = tf.convert_to_tensor(arr)
    with tf.GradientTape() as tape:
        tape.watch(tensor)
        preds = model(tensor)
        top = preds[:, tf.argmax(preds[0])]
    sal = tf.reduce_max(tf.abs(tape.gradient(top, tensor)[0]), axis=-1).numpy()
    sal = np.uint8(255 * (sal - sal.min()) / (sal.max() - sal.min() + 1e-8))

    _, mask = cv2.threshold(sal, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawn = np.uint8(arr[0] * 255).copy()
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cv2.rectangle(drawn, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.figure(figsize=(6, 6))
    plt.imshow(drawn)
    plt.axis("off")
    plt.title("Approximate object location (saliency)")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train the waste classifier.")
    parser.add_argument("--data", default="dataset_split", help="Folder containing train/ and test/")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--out", default="best_model.h5", help="Where to save the best model")
    parser.add_argument("--weights", default="imagenet", help="'imagenet' or 'none' (random init)")
    args = parser.parse_args()

    print("GPUs available:", len(tf.config.list_physical_devices("GPU")))
    train_gen, val_gen, test_gen = build_generators(args.data)
    model = build_model(None if args.weights == "none" else args.weights)
    model.summary()

    model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=[
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint(args.out, save_best_only=True),
        ],
    )

    # Final, one-time evaluation on the held-out test set
    test_loss, test_acc = model.evaluate(test_gen)
    y_pred = np.argmax(model.predict(test_gen), axis=1)
    y_true = test_gen.classes
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))

    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump({"test_accuracy": test_acc, "test_loss": test_loss, "per_class": report}, f, indent=2)

    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=CLASSES)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion matrix (test set)")
    plt.savefig("results/confusion_matrix.png", bbox_inches="tight")
    plt.close()

    saliency_box(model, test_gen.filepaths[0], "results/saliency_example.png")
    print("Saved model to", args.out, "and metrics/plots to results/")


if __name__ == "__main__":
    main()
