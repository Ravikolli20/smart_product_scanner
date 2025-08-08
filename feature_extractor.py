# feature_extractor.py
# Description: Handles loading the ML model and extracting features from images.

import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Import configuration
from config import IMG_SIZE

@st.cache_resource
def load_model():
    """
    Loads the pre-trained MobileNetV2 model, modified to output feature vectors.
    The model is cached using Streamlit's caching to prevent reloading on every run.
    """
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False, # Exclude the final classification layer
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    # Define the model to use the base model's input and the global average pooling output
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d').output)
    return model

def _preprocess_image(img_path_or_buffer):
    """
    Internal function to load and preprocess an image for the MobileNetV2 model.
    """
    img = image.load_img(img_path_or_buffer, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def extract_features(model, img_path_or_buffer):
    """
    Extracts a feature vector (embedding) from a preprocessed image using the model.
    """
    preprocessed_img = _preprocess_image(img_path_or_buffer)
    features = model.predict(preprocessed_img, verbose=0) # Set verbose=0 to hide prediction progress bar
    # Normalize the feature vector for effective cosine similarity comparison
    normalized_features = features / np.linalg.norm(features)
    return normalized_features.flatten()