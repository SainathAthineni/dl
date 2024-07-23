import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

# Constants
IMG_SIZE = 224
CATEGORIES = ['ahegao', 'angry', 'happy', 'neutral', 'sad', 'surprise']

# Function to preprocess the image
def preprocess_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Histogram equalization
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.equalizeHist(img)
    else:
        for i in range(img.shape[2]):
            img[:, :, i] = cv2.equalizeHist(img[:, :, i])

    # Intensity thresholding
    lower_threshold = 30
    upper_threshold = 200
    _, img = cv2.threshold(img, lower_threshold, upper_threshold, cv2.THRESH_TOZERO)

    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Load the model
model = load_model("cnn_model.h5")

# Streamlit UI
st.title("Facial Expression Recognition")
st.write("Upload an image to predict the facial expression.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Read the image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Display the uploaded image
    st.image(image, channels="BGR", caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make prediction
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions)
    predicted_label = CATEGORIES[predicted_class]

    # Display the prediction
    st.write(f"Predicted Facial Expression: **{predicted_label.capitalize()}**")