import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("/Users/rohankumbhar/python/best_model.h5")

# Define class labels (make sure the order matches your training dataset)
class_labels = ["metal", "plastic", "Paper"]

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access camera")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame")
        break

    # Crop to center square for better object focus
    h, w, _ = frame.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped = frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

    # Apply Gaussian blur to reduce noise
    cropped = cv2.GaussianBlur(cropped, (5, 5), 0)

    # Resize and preprocess the frame
    img = cv2.resize(cropped, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # Predict the class
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index] * 100

    # Debug: print confidence for all classes
    print("Class Probabilities:")
    for i, prob in enumerate(prediction[0]):
        print(f"{class_labels[i]}: {prob * 100:.2f}%")

    # Display prediction only if confidence is high
    if confidence > 60:
        label = class_labels[class_index]
        display_text = f"{label} ({confidence:.2f}%)"
        # Draw red rectangle on the cropped area
        cv2.rectangle(frame, (start_x, start_y), (start_x + min_dim, start_y + min_dim), (0, 0, 255), 2)
        # Put label near the rectangle
        cv2.putText(frame, display_text, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        display_text = "Uncertain"

    # Show prediction on the frame
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Waste Classification - Press 'q' to Quit", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()