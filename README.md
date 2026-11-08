# AI Waste Segregation

An image classifier that sorts waste into **metal**, **plastic** and **paper**, built for automatic segregation at the point of disposal. It fine-tunes a MobileNetV2 backbone, exports a quantised TFLite model (2.7 MB) for edge devices, and includes a live webcam demo.

## How it works

```
raw images ──► preprocess.py ──► split_dataset.py ──► train_model.py ──► best_model.h5 ──► convert_to_tflite.py ──► model.tflite
 (per class)    resize, drop      80/20 train/test     MobileNetV2 fine-tune               dynamic-range quantisation
                unreadable files  (fixed seed)         + held-out validation
```

- **Model:** MobileNetV2 pretrained on ImageNet; the first 100 layers are frozen and the rest are fine-tuned, with a 128-unit dense head and dropout 0.5.
- **Training:** heavy augmentation (rotation, shift, shear, zoom, flip, brightness), Adam at 1e-4, and early stopping on a validation set held out from the training images. The test set is used once, at the end.
- **Outputs:** test accuracy, a per-class precision/recall report, a confusion matrix and a saliency-based localisation example, saved to `results/`.
- **Deployment:** `model.tflite` is about 3.5× smaller than the float model and runs on CPU-only and embedded hardware.

All scripts share `config.py`, which defines the class order and pixel scaling, so inference always matches training.

## Results

<!-- TODO: fill in from results/metrics.json after training -->
| Metric | Value |
|---|---|
| Test accuracy | _TBD_ |
| Dataset size | _TBD images (metal / plastic / paper)_ |

<!-- After training, commit results/ and uncomment:
![Confusion matrix](results/confusion_matrix.png)
-->

## Getting started

```bash
pip install -r requirements.txt
```

**Classify an image** with the TFLite model:

```bash
python inference.py path/to/image.jpg
```

**Live webcam demo** (press `q` to quit):

```bash
python camera_test.py
```

**Train from scratch** on your own images, placed in `dataset_raw/metal`, `dataset_raw/plastic` and `dataset_raw/paper`:

```bash
python preprocess.py        # dataset_raw/ -> dataset_preprocessed/
python split_dataset.py     # -> dataset_split/train, dataset_split/test
python train_model.py       # -> best_model.h5, results/
python convert_to_tflite.py # -> model.tflite
```

## Project structure

| File | Purpose |
|---|---|
| `config.py` | Class order, image size and shared preprocessing |
| `model.py` | Model architecture and a loader that works on Keras 2 and 3 |
| `preprocess.py` | Resize images and skip unreadable files |
| `split_dataset.py` | Reproducible 80/20 train/test split |
| `train_model.py` | Training, evaluation and result plots |
| `convert_to_tflite.py` | Export to quantised TFLite |
| `inference.py` | Single-image prediction with TFLite |
| `camera_test.py` | Real-time webcam classification |

## Limitations and next steps

- Three classes only; glass, organic and e-waste are natural additions.
- One object per frame. The saliency box gives a rough location but isn't a detector; a lightweight detector such as YOLO would handle mixed piles.
- Deploying `model.tflite` on a Raspberry Pi with a servo-driven sorting flap would complete the segregation loop.

## Author

**Rohan Kumbhar**, B.Tech Electronics and Computer Engineering, Ramrao Adik Institute of Technology, Navi Mumbai
