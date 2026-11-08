"""Live webcam classification. Press 'q' to quit.

Usage:
    python camera_test.py [--model best_model.h5] [--camera 0] [--threshold 60]
"""
import argparse

import cv2
import numpy as np

from config import CLASSES, preprocess
from model import load_trained_model


def main():
    parser = argparse.ArgumentParser(description="Classify waste from a webcam feed.")
    parser.add_argument("--model", default="best_model.h5", help="Path to the Keras model")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--threshold", type=float, default=60.0,
                        help="Minimum confidence (%%) before a label is shown")
    args = parser.parse_args()

    model = load_trained_model(args.model)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Error: cannot access camera")

    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to grab frame")
            break

        # Crop to the centre square so the object fills the model's view
        h, w, _ = frame.shape
        side = min(h, w)
        x0, y0 = (w - side) // 2, (h - side) // 2
        crop = cv2.GaussianBlur(frame[y0:y0 + side, x0:x0 + side], (5, 5), 0)

        # OpenCV frames are BGR; the model was trained on RGB images
        batch = preprocess(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        probs = model.predict(batch, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = probs[idx] * 100

        if confidence > args.threshold:
            text = f"{CLASSES[idx]} ({confidence:.1f}%)"
            cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), (0, 0, 255), 2)
            cv2.putText(frame, text, (x0, max(y0 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            text = "Uncertain"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Waste Classification - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
