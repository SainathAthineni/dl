import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = load_model('C:/Users/saina/Downloads/vggunfr_model.h5')

# Define categories
categories = ['ahegao', 'angry', 'happy', 'neutral', 'sad', 'surprise']

def predict_image(model, image_path, categories):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = categories[np.argmax(predictions)]
    
    return predicted_class

# Streamlit app
st.title("Facial Expression Recognition")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Save the uploaded file temporarily
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Predict
    predicted_class = predict_image(model, "temp.jpg", categories)
    
    st.write(f"Prediction: {predicted_class}")
