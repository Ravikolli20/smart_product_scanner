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

    # --- INITIALIZE SESSION STATE ---
    # This is crucial for conditionally rendering the camera widget.
    if 'show_camera' not in st.session_state:
        st.session_state.show_camera = False

    # --- LOAD MODEL AND DATA ---
    model = load_model()

    # This logic is non-destructive.
    if not os.path.exists(METADATA_FILE):
        generate_mock_dataset()
    if not os.path.exists(EMBEDDINGS_FILE):
        generate_embeddings(model)

    # Load the pre-computed embeddings and product metadata
    all_features = np.load(EMBEDDINGS_FILE)
    with open(METADATA_FILE, 'r') as f:
        product_metadata = json.load(f)

    # --- USER INPUT ---
    st.sidebar.header("Choose Input Method")
    
    # Use columns for a cleaner layout
    col1, col2 = st.sidebar.columns(2)

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

    with col2:
        # This button will set the session state to show the camera
        if st.button("📸 Use Camera"):
            st.session_state.show_camera = True

    # --- CONDITIONAL CAMERA INPUT ---
    camera_input = None
    if st.session_state.show_camera:
        camera_input = st.camera_input(
            "Take a picture of the product.",
            label_visibility="collapsed"
        )
        st.sidebar.info(
            "We need camera access for a one-time photo. "
            "Your image is not processed or stored on our servers."
        )

    # Determine which input was used
    input_image = uploaded_file or camera_input

    # --- PROCESSING AND DISPLAYING RESULTS ---
    if input_image is not None:
        # Important: Turn off the camera view once an image is captured or uploaded
        st.session_state.show_camera = False

        st.subheader("Your Query Image")
        query_image_display = Image.open(input_image)
        st.image(query_image_display, width=250, caption="This is the product you're looking for.")

        with st.spinner("Analyzing image and finding similar products..."):
            input_features = extract_features(model, input_image)
            similar_products = find_similar_products(input_features, all_features, product_metadata)

            st.subheader(f"Top {NUM_SIMILAR_PRODUCTS} Similar Products Found")
            st.markdown("---")

            cols = st.columns(NUM_SIMILAR_PRODUCTS)
            for i, (product, score) in enumerate(similar_products[:NUM_SIMILAR_PRODUCTS]):
                with cols[i]:
                    st.image(product['image_path'], caption=f"Similarity: {score:.2%}", use_container_width=True)
                    st.markdown(f"**{product['name']}**")
                    st.markdown(f"**Price:** ${product['price']:.2f}")
    else:
        if not st.session_state.show_camera:
             st.info("Please upload an image or use the camera to start.")

if __name__ == '__main__':
    main()
