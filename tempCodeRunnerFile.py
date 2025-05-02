import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import cv2

# Check if TensorFlow is using GPU
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Set paths to training and testing datasets
train_dir = "/Users/rohankumbhar/dataset_split/train"
test_dir = "/Users/rohankumbhar/dataset_split/test"

# Define Image Parameters
IMG_SIZE = (224, 224)  # Resize images to 224x224
BATCH_SIZE = 32

# Enhanced Data Augmentation for Training Data
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,     
    rotation_range=30,    
    width_shift_range=0.3, 
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,        
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # Vary brightness
    fill_mode='nearest'
)

# No Augmentation for Testing Data (Only Normalization)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Explicitly set the class order
classes = ['metal', 'plastic', 'paper']

# Load Training Data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=classes
)
print("Class Indices:", train_generator.class_indices)  # Expected: {'metal': 0, 'plastic': 1, 'paper': 2}
  
# Load Testing Data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=classes
)

# Load Pretrained MobileNetV2 Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze Pretrained Layers
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Build Model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 output classes: Metal, Plastic, Paper
])

# Print model summary
model.summary()

# Compile the Model with Lower Learning Rate
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Reduced from 0.001 to 0.0001
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the Model with More Epochs
EPOCHS = 25  

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("best_model.h5", save_best_only=True)
]

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=callbacks
)

# Evaluate the Model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Generate Confusion Matrix
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# Get class labels in the same order
class_labels = list(test_generator.class_indices.keys())

# Compute and plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
plt.figure(figsize=(6, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()

# Save the Model
model.save("/Users/rohankumbhar/python/waste_classifier.h5")
print("Model saved successfully!")

# Visualize the region influencing the prediction
img_path = test_generator.filepaths[0]
img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Use GradientTape to get gradients of the top predicted class
img_tensor = tf.convert_to_tensor(img_array)
with tf.GradientTape() as tape:
    tape.watch(img_tensor)
    preds = model(img_tensor)
    top_pred_index = tf.argmax(preds[0])
    top_class_channel = preds[:, top_pred_index]

grads = tape.gradient(top_class_channel, img_tensor)[0]
saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()

# Normalize and convert to uint8
saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
saliency = np.uint8(saliency * 255)

# Threshold and find contours
_, binary_map = cv2.threshold(saliency, 100, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding box
img_to_draw = np.uint8(img_array[0] * 255).copy()
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(img_to_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show result
plt.figure(figsize=(6, 6))
plt.imshow(img_to_draw)
plt.axis('off')
plt.title("Approximate Object Detection via Saliency Map")
plt.show()