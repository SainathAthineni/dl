

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


# Correct file paths
generator_path = "C:/Users/saina/Downloads/generator.h5"
discriminator_path = "C:/Users/saina/Downloads/discriminator.h5"
cgan_path = "C:/Users/saina/Downloads/cgan.h5"

# Load the trained generator model
generator = load_model(generator_path)

# Define categories
categories = ['Angry', 'Happy', 'Sad', 'Neutral', 'Surprise', 'Ahegao']

# Set latent dimension and number of classes
latent_dim = 100
num_classes = len(categories)

# Streamlit interface
st.title("Face Expression Generator using cGAN")

expression = st.selectbox("Select an expression", categories)
if st.button("Generate Face"):
    noise = np.random.normal(0, 1, (1, latent_dim))
    label = np.array([categories.index(expression)]).reshape(-1, 1)
    gen_img = generator.predict([noise, label])
    gen_img = 0.5 * gen_img + 0.5

    fig, ax = plt.subplots()
    ax.imshow(gen_img[0])
    ax.axis('off')
    st.pyplot(fig)

