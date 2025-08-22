# data_setup.py
# Description: Functions to generate the mock dataset and pre-compute embeddings.

import os
import json
import numpy as np
import streamlit as st
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# Import project modules
from config import IMAGE_DIR, METADATA_FILE, EMBEDDINGS_FILE, IMG_SIZE
# --- START: CORRECTION ---
# Import the new, dedicated function for database creation
from feature_extractor import extract_features_for_database
# --- END: CORRECTION ---

def _create_placeholder_image(path, text, size):
    """Generates a placeholder image with text."""
    try:
        img = Image.new('RGB', size, color = (235, 244, 250))
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        position = ((size[0]-text_width)/2, (size[1]-text_height)/2)
        
        draw.text(position, text, fill=(74, 74, 74), font=font)
        img.save(path)
    except Exception as e:
        st.error(f"Failed to create placeholder image {path}: {e}")

def generate_mock_dataset():
    """
    Generates a mock dataset of products by creating images locally.
    """
    st.info("First-time setup: Generating mock dataset locally...")
    os.makedirs(IMAGE_DIR, exist_ok=True)

    products = [
        {"id": "shoe001", "name": "Running Shoe", "category": "Shoes", "price": 59.99, "product_url": "https://example.com/product/shoe001"},
        {"id": "watch001", "name": "Classic Chronograph", "category": "Watches", "price": 250.00, "product_url": "https://example.com/product/watch001"},
    ]

    for product in tqdm(products, desc="Generating local images"):
        img_path = os.path.join(IMAGE_DIR, f"{product['id']}.jpg")
        product["image_path"] = img_path
        if not os.path.exists(img_path):
            _create_placeholder_image(img_path, product['name'], IMG_SIZE)

    with open(METADATA_FILE, 'w') as f:
        json.dump(products, f, indent=4)
    st.success("Mock dataset created.")

def generate_embeddings(model):
    """
    Generates and saves feature embeddings for all products in the dataset.
    """
    with st.spinner("Generating feature embeddings for the product database..."):
        with open(METADATA_FILE, 'r') as f:
            products = json.load(f)

        all_features = []
        for product in tqdm(products, desc="Generating embeddings"):
            if os.path.exists(product["image_path"]):
                # --- START: CORRECTION ---
                # Call the new function that doesn't require the YOLO model
                features = extract_features_for_database(model, product["image_path"])
                # --- END: CORRECTION ---
                
                # Ensure features were extracted successfully before appending
                if features is not None:
                    all_features.append(features)

        np.save(EMBEDDINGS_FILE, np.array(all_features))
    st.success("Embeddings database generated successfully!")
