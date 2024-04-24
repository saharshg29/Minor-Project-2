import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

IMG_SIZE = 224

# Load the saved model
model = load_model('diabetic_retinopathy_model.h5')

# Load the image
image = cv2.imread("./2.png")

# Preprocess the image similar to training data
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize to 224x224
image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), IMG_SIZE/10), -4, 128)  # Apply filter

# Convert image to numpy array and add channel dimension
image_np = np.expand_dims(image, axis=-1)  # Add channel dimension
image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension

# Make predictions
predictions = model.predict(image_np)

# Get the maximum prediction value and its corresponding index
max_prediction = np.max(predictions)
max_index = np.argmax(predictions)

# You can then use these predictions as needed
print("Maximum Prediction Value:", max_prediction)
print("Predicted Class:", max_index)
# print("All Predictions:", predictions)

