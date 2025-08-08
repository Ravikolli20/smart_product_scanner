# data_setup.py
# Description: Functions to generate the mock dataset and pre-compute embeddings.

import os
import json
import numpy as np
import streamlit as st
import urllib.request
from tqdm import tqdm # For showing progress bar in console

# Import project modules
from config import IMAGE_DIR, METADATA_FILE, EMBEDDINGS_FILE
from feature_extractor import extract_features

def generate_mock_dataset():
    """
    Generates a mock dataset of products if it doesn't already exist.
    """
    st.info("First-time setup: Generating mock dataset...")
    os.makedirs(IMAGE_DIR, exist_ok=True)

    products = [
        {"id": "shoe001", "name": "Running Shoe", "category": "Shoes", "price": 59.99},
        {"id": "shoe002", "name": "Casual Sneaker", "category": "Shoes", "price": 75.00},
        {"id": "shoe003", "name": "Hiking Boot", "category": "Shoes", "price": 120.50},
        {"id": "shoe004", "name": "Formal Oxford", "category": "Shoes", "price": 99.99},
        {"id": "bag001", "name": "Leather Backpack", "category": "Bags", "price": 89.99},
        {"id": "bag002", "name": "Canvas Tote", "category": "Bags", "price": 25.00},
        {"id": "bag003", "name": "Travel Duffel", "category": "Bags", "price": 65.50},
        {"id": "bag004", "name": "Laptop Messenger", "category": "Bags", "price": 45.00},
        {"id": "watch001", "name": "Classic Chronograph", "category": "Watches", "price": 250.00},
        {"id": "watch002", "name": "Digital Sport Watch", "category": "Watches", "price": 49.99},
        {"id": "watch003", "name": "Minimalist Analog", "category": "Watches", "price": 150.00},
        {"id": "watch004", "name": "Smart Watch", "category": "Watches", "price": 300.00},
        {"id": "chair001", "name": "Ergonomic Office Chair", "category": "Furniture", "price": 220.00},
        {"id": "chair002", "name": "Modern Accent Chair", "category": "Furniture", "price": 180.75},
        {"id": "chair003", "name": "Wooden Dining Chair", "category": "Furniture", "price": 75.00},
        {"id": "chair004", "name": "Lounge Recliner", "category": "Furniture", "price": 350.00},
    ]

    for product in tqdm(products, desc="Downloading images"):
        img_path = os.path.join(IMAGE_DIR, f"{product['id']}.jpg")
        product["image_path"] = img_path
        if not os.path.exists(img_path):
            url = f"https://placehold.co/224x224/EBF4FA/4A4A4A?text={product['name'].replace(' ', '+')}"
            urllib.request.urlretrieve(url, img_path)

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
            features = extract_features(model, product["image_path"])
            all_features.append(features)

        np.save(EMBEDDINGS_FILE, np.array(all_features))
    st.success("Embeddings database generated successfully!")