import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the model with the custom name
model = load_model('C:/Users/saina/Downloads/vggadd_model.h5')

# Define the categories
categories = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Define preprocess function
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    _, img = cv2.threshold(img, 30, 200, cv2.THRESH_TOZERO)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img_to_array(img) / 255.0
    return img

# Streamlit interface
st.title("Facial Expression Recognition")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    st.image(image, channels="BGR")

    # Preprocess the image
    img = preprocess_image(image)

    # Predict the category
    pred = model.predict(np.expand_dims(img, axis=0))
    pred_class = categories[np.argmax(pred)]

    # Display the prediction
    st.write(f"Predicted Expression: {pred_class}")
