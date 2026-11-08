"""Classify a single image with the TFLite model.

Usage:
    python inference.py path/to/image.jpg [--model model.tflite]
"""
import argparse

import numpy as np
from PIL import Image

try:  # LiteRT replaces tf.lite.Interpreter in TensorFlow 2.20+
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter

from config import CLASSES, preprocess


def main():
    parser = argparse.ArgumentParser(description="Classify a waste image as metal, plastic or paper.")
    parser.add_argument("image", help="Path to the image to classify")
    parser.add_argument("--model", default="model.tflite", help="Path to the TFLite model")
    args = parser.parse_args()

    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    interpreter.set_tensor(input_details[0]["index"], preprocess(rgb))
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]["index"])[0]

    predicted = int(np.argmax(probs))
    print(f"Prediction: {CLASSES[predicted]} ({probs[predicted] * 100:.1f}%)")
    for name, p in zip(CLASSES, probs):
        print(f"  {name:<8} {p * 100:5.1f}%")


if __name__ == "__main__":
    main()
