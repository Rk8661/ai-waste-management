import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('best_model.h5')

# Export the model to SavedModel format
model.export('saved_model')  # Correct export method for TFLite

# Convert the SavedModel to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()

# Save the TFLite model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved as model.tflite")