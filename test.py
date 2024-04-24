import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

IMG_SIZE = 224

# Load the saved model
model = load_model('diabetic_retinopathy_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize to 224x224
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), IMG_SIZE/10), -4, 128)  # Apply filter
    image_np = np.expand_dims(image, axis=-1)  # Add channel dimension
    image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension
    return image_np

# def preprocess_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to 224x224
    resized_image = cv2.resize(gray_image, (IMG_SIZE, IMG_SIZE))
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(resized_image)
    
    # Normalize the image
    normalized_image = clahe_image / 255.0
    
    # Add channel dimension and batch dimension
    image_np = np.expand_dims(normalized_image, axis=-1)
    image_np = np.expand_dims(image_np, axis=0)
    
    return image_np


# Function to make predictions
def predict(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    max_prediction = np.max(predictions)
    max_index = np.argmax(predictions)
    return max_prediction, max_index

# Streamlit app
def main():
    st.title("Diabetic Retinopathy Prediction App")
    st.write("Upload an image of an eye to predict the presence of diabetic retinopathy.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction on the uploaded image
        if st.button("Predict"):
            max_prediction, max_index = predict(image)
            st.write(f"Maximum Prediction Value: {max_prediction}")
            st.write(f"Predicted Class: {max_index}")

if __name__ == "__main__":
    main()
