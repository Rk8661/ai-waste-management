"""Shared settings for training and inference.

Every script imports CLASSES and preprocess() from here, so the class order
and pixel scaling used at inference time always match what the model was
trained with.
"""
import numpy as np

# Index order of the model's softmax output. The shipped model was trained
# with this order; changing it requires retraining.
CLASSES = ["metal", "plastic", "paper"]

IMG_SIZE = (224, 224)


def preprocess(rgb_image):
    """Turn an RGB image (H x W x 3, uint8) into a 1 x 224 x 224 x 3 float32
    batch scaled to [0, 1], matching `rescale=1/255` used during training."""
    from PIL import Image

    img = Image.fromarray(np.asarray(rgb_image, dtype=np.uint8)).resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)
