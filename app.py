import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# Load your trained model
@st.cache_resource
def load_vgg_model():
    model = load_model('/Users/prabinawal/Downloads/streamlit_app/best_model.h5', compile=False)
    return model

model = load_vgg_model()

# Title of the app
st.title('Face Expression Identification/Generation App')

# File uploader for the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image
        img = image.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict the expression
        predictions = model.predict(img)
        st.write(predictions)
        
        # Convert predictions to class labels
        class_names = ['Happy', 'Sad', 'Surprise', 'Angry']  # Adjust based on your classes
        predicted_class = class_names[np.argmax(predictions)]
        st.write(f'Predicted Expression: {predicted_class}')
    except Exception as e:
        st.error(f"An error occurred: {e}")


