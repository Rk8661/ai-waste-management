"""Model architecture and a loader that works across Keras versions."""
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

from config import CLASSES, IMG_SIZE


def build_model(weights="imagenet"):
    """MobileNetV2 backbone (top layers fine-tuned) + small classification head."""
    base = MobileNetV2(weights=weights, include_top=False, input_shape=IMG_SIZE + (3,))
    base.trainable = True
    for layer in base.layers[:100]:
        layer.trainable = False

    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(len(CLASSES), activation="softmax"),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def load_trained_model(path="best_model.h5"):
    """Load a saved model. Keras 3 can't deserialise a Sequential model that
    wraps MobileNetV2 from .h5, so fall back to rebuilding the architecture
    and loading only the weights."""
    try:
        return load_model(path, compile=False)
    except (ValueError, TypeError):
        model = build_model(weights=None)
        model.load_weights(path)
        return model
