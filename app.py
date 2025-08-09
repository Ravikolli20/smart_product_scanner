# app.py
# Description: The main Streamlit web application file.

import os
import json
import numpy as np
from PIL import Image
import streamlit as st

# Import project modules
from config import METADATA_FILE, EMBEDDINGS_FILE, NUM_SIMILAR_PRODUCTS
from data_setup import generate_mock_dataset, generate_embeddings
from feature_extractor import load_model, extract_features
from similarity import find_similar_products

def main():
    """
    Main function to run the Streamlit web application.
    """
    st.set_page_config(page_title="Smart Product Scanner", layout="wide")

    # --- HEADER ---
    st.title("🛍️ Smart Product Scanner")
    st.markdown("Upload or take a picture of a product to find similar items in our catalog.")

    # --- LOAD MODEL AND DATA ---
    model = load_model()

    # --- START: CORRECTION ---
    # This logic is now non-destructive.
    # Step 1: Create the default dataset if the JSON file doesn't exist.
    if not os.path.exists(METADATA_FILE):
        generate_mock_dataset()

    # Step 2: Generate embeddings if the embeddings file doesn't exist.
    # This allows users to add to products.json and just delete embeddings.npy to update.
    if not os.path.exists(EMBEDDINGS_FILE):
        generate_embeddings(model)
    # --- END: CORRECTION ---

    # Load the pre-computed embeddings and product metadata
    all_features = np.load(EMBEDDINGS_FILE)
    with open(METADATA_FILE, 'r') as f:
        product_metadata = json.load(f)

    # --- USER INPUT ---
    st.sidebar.header("Upload or Capture Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    camera_input = st.sidebar.camera_input("...or use your camera")

    input_image = uploaded_file or camera_input

    # --- PROCESSING AND DISPLAYING RESULTS ---
    if input_image is not None:
        # Display the user's query image
        st.subheader("Your Query Image")
        query_image_display = Image.open(input_image)
        st.image(query_image_display, width=250, caption="This is the product you're looking for.")

        # Find and display similar products
        with st.spinner("Analyzing image and finding similar products..."):
            input_features = extract_features(model, input_image)
            similar_products = find_similar_products(input_features, all_features, product_metadata)

            st.subheader(f"Top {NUM_SIMILAR_PRODUCTS} Similar Products Found")
            st.markdown("---")

            # Display results in columns
            cols = st.columns(NUM_SIMILAR_PRODUCTS)
            for i, (product, score) in enumerate(similar_products[:NUM_SIMILAR_PRODUCTS]):
                with cols[i]:
                    st.image(product['image_path'], caption=f"Similarity: {score:.2%}", use_container_width=True)
                    st.markdown(f"**{product['name']}**")
                    st.markdown(f"**Price:** ${product['price']:.2f}")
    else:
        st.info("Please upload an image or use the camera to start.")

if __name__ == '__main__':
    main()
