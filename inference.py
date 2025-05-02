import numpy as np
import tensorflow as tf
from PIL import Image

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels
class_names = ['metal', 'paper', 'plastic']  # Make sure the order matches your training

# Load and preprocess image
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img.show()  # Show the image to verify it's correct
    img = img.resize((224, 224))  # Resize to model's expected size
    img = np.array(img) / 255.0   # Normalize pixel values
    img = np.expand_dims(img, axis=0).astype(np.float32)  # Add batch dimension
    return img

# Path to test image
image_path = "/Users/rohankumbhar/dataset2/plastic/plastic (544).jpg"

# Run inference
input_data = load_image(image_path)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Get predicted label
predicted_index = np.argmax(output_data)
predicted_label = class_names[predicted_index]

print(f"🔍 Prediction: {predicted_label}")