"""Convert the trained Keras model to a quantised TFLite model.

Usage:
    python convert_to_tflite.py [--model best_model.h5] [--out model.tflite]
"""
import argparse

import tensorflow as tf

from model import load_trained_model


def main():
    parser = argparse.ArgumentParser(description="Export the Keras model to TFLite.")
    parser.add_argument("--model", default="best_model.h5")
    parser.add_argument("--out", default="model.tflite")
    args = parser.parse_args()

    model = load_trained_model(args.model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic-range quantisation
    with open(args.out, "wb") as f:
        f.write(converter.convert())
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
