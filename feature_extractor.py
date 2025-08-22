# feature_extractor.py
# Description: Handles loading ML models and extracting features from images.
# Now includes a YOLOv8 model for object detection and cropping.

import numpy as np
import streamlit as st
from PIL import Image
import cv2
# --- START: CORRECTION ---
from ultralytics import YOLO
# --- END: CORRECTION ---
import tensorflow as tf

# Keras imports for the similarity model
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing import image as keras_image
from keras.models import Model

# Import configuration
from config import IMG_SIZE

FINE_TUNED_MODEL_PATH = 'fine_tuned_model.keras'

@st.cache_resource
def load_mobilenet_model():
    """
    Loads our custom fine-tuned Keras model.
    If it doesn't exist, it falls back to the generic MobileNetV2.
    """
    try:
        print("Loading fine-tuned model...")
        model = tf.keras.models.load_model(FINE_TUNED_MODEL_PATH)
        feature_extractor_layer = model.get_layer('global_average_pooling2d')
        feature_extractor_model = Model(
            inputs=model.input,
            outputs=feature_extractor_layer.output
        )
        return feature_extractor_model
    except Exception as e:
        print(f"Could not load fine-tuned model: {e}. Falling back to generic MobileNetV2.")
        from keras.applications.mobilenet_v2 import MobileNetV2
        from keras.layers import GlobalAveragePooling2D
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        model = Model(inputs=base_model.input, outputs=x)
        return model

@st.cache_resource
def load_yolo_model():
    """Loads the pre-trained YOLOv8 model for object detection."""
    model = YOLO('yolov8n.pt')
    return model

def crop_main_object(yolo_model, pil_image):
    """
    Detects objects in an image using YOLO, finds the largest one,
    and returns the cropped image and the bounding box coordinates.
    """
    cv_image = np.array(pil_image)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    results = yolo_model.predict(cv_image, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    if len(boxes) == 0:
        return pil_image, None

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    largest_box_index = np.argmax(areas)
    x1, y1, x2, y2 = map(int, boxes[largest_box_index])
    cropped_pil_image = pil_image.crop((x1, y1, x2, y2))
    bounding_box = (x1, y1, x2, y2)
    
    return cropped_pil_image, bounding_box

def _preprocess_image_for_mobilenet(pil_image):
    """
    Internal function to preprocess a PIL image for the MobileNetV2 model.
    """
    img_resized = pil_image.resize(IMG_SIZE)
    img_array = keras_image.img_to_array(img_resized)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def extract_features_for_database(mobilenet_model, image_path):
    """
    Extracts features from a clean product image file for database creation.
    """
    try:
        pil_image = Image.open(image_path).convert("RGB")
        preprocessed_img = _preprocess_image_for_mobilenet(pil_image)
        features = mobilenet_model.predict(preprocessed_img, verbose=0)
        normalized_features = features / np.linalg.norm(features)
        return normalized_features.flatten()
    except Exception:
        return None

def extract_features(mobilenet_model, yolo_model, pil_image):
    """
    Extracts a feature vector from a user-provided image.
    1. Detects and crops the main object using YOLO.
    2. Extracts features from the cropped object using MobileNetV2.
    """
    cropped_image, bounding_box = crop_main_object(yolo_model, pil_image)
    preprocessed_img = _preprocess_image_for_mobilenet(cropped_image)
    features = mobilenet_model.predict(preprocessed_img, verbose=0)
    normalized_features = features / np.linalg.norm(features)
    
    return normalized_features.flatten(), bounding_box
