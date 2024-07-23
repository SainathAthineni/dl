import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the trained models from the SavedModel format
encoder = tf.saved_model.load('C:/Users/saina/OneDrive/Documents/encoder.json')
decoder = tf.saved_model.load('C:/Users/saina/OneDrive/Documents/decoder.json')

# Define latent dimension and categories
latent_dim = 100
categories = ['Angry', 'Happy', 'Sad', 'Neutral', 'Surprise', 'Ahegao']
num_classes = len(categories)

# Streamlit UI
st.title("VAE Face Generator")

st.sidebar.header("Generate Faces")

if st.sidebar.button("Generate Faces"):
    # Ensure the latent vectors are of type tf.float32
    random_latent_vectors = np.random.normal(size=(num_classes, latent_dim)).astype(np.float32)
    generated_images = decoder(random_latent_vectors, training=False)

    fig, axes = plt.subplots(1, num_classes, figsize=(20, 4))
    for i in range(num_classes):
        axes[i].imshow(generated_images[i].numpy())
        axes[i].set_title(categories[i])
        axes[i].axis('off')
    st.pyplot(fig)

st.write("This application uses a Variational Autoencoder (VAE) to generate faces corresponding to different categories.")
