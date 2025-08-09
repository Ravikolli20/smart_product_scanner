# app.py
# Smart Product Scanner — working Light/Dark toggle
import os
import json
import base64
import numpy as np
from PIL import Image
import streamlit as st

# Your project modules (unchanged)
from config import METADATA_FILE, EMBEDDINGS_FILE, NUM_SIMILAR_PRODUCTS
from data_setup import generate_mock_dataset, generate_embeddings
from feature_extractor import load_model, extract_features
from similarity import find_similar_products


def image_to_base64(path):
    """Converts a local image file to a base64 string for embedding in HTML."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _get_theme_values(theme: str):
    """Return palette values for the given theme."""
    if theme.lower() == "dark":
        return {
            "gradient_bg": "linear-gradient(135deg, #0f2027, #203a43, #2c5364)",
            "text_color": "#e8eef1",
            "card_bg": "#1e1f22",
            "muted_text": "#bfc7cc",
            "accent": "#6a11cb",
            "hero_gradient": "linear-gradient(90deg, #141e30, #243b55)"
        }
    else:
        return {
            "gradient_bg": "linear-gradient(90deg, #f8f9fa, #eef2f3)",
            "text_color": "#0b1321",
            "card_bg": "#ffffff",
            "muted_text": "#57606a",
            "accent": "#2575fc",
            "hero_gradient": "linear-gradient(90deg, #6a11cb, #2575fc)"
        }


def inject_css(theme_vals: dict):
    """Inject CSS tailored to the selected theme."""
    css = f"""
    <style>
    .stApp {{
        background: {theme_vals['gradient_bg']} !important;
        color: {theme_vals['text_color']} !important;
    }}
    .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }}
    .hero {{
        text-align: center;
        padding: 1.4rem;
        background: {theme_vals['hero_gradient']} !important;
        color: white !important;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }}
    .card {{
        background-color: {theme_vals['card_bg']} !important;
        color: {theme_vals['text_color']} !important;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 6px 18px rgba(2,6,23,0.10) !important;
    }}
    .product-card {{
        background-color: {theme_vals['card_bg']} !important;
        border-radius: 12px;
        padding: 0.9rem;
        box-shadow: 0 6px 18px rgba(2,6,23,0.08) !important;
        text-align: center;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }}
    .product-card:hover {{
        transform: translateY(-6px) scale(1.02);
        box-shadow: 0 12px 28px rgba(2,6,23,0.12) !important;
    }}
    .product-image {{
        border-radius: 8px;
        width: 100%;
        height: 160px;
        object-fit: contain;
        margin-bottom: 0.7rem;
    }}
    .product-title {{
        font-weight: 700;
        font-size: 1rem;
        color: {theme_vals['text_color']} !important;
        margin-bottom: 0.25rem;
        min-height: 2.2rem;
    }}
    .product-price {{
        color: {theme_vals['accent']} !important;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }}
    .query-card {{
        background: {theme_vals['card_bg']} !important;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(2,6,23,0.06) !important;
        text-align: center;
        margin-bottom: 1rem;
    }}
    .muted {{
        color: {theme_vals['muted_text']} !important;
    }}
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        color: {theme_vals['text_color']} !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Smart Product Scanner", layout="wide")

    if "theme" not in st.session_state:
        st.session_state["theme"] = "Light"

    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.radio("Theme", options=["Light", "Dark"], key="theme",
                 index=0 if st.session_state["theme"] == "Light" else 1)
        st.markdown("---")
        st.markdown("### 📷 Select Input Method")
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        take_photo = st.button("📸 Take a Photo")
        if take_photo:
            st.session_state["show_camera"] = not st.session_state.get("show_camera", False)
        camera_input = None
        if st.session_state.get("show_camera", False):
            camera_input = st.camera_input("Capture Product Image")
            st.info("Your photo will be used only for analysis.")

    theme_choice = st.session_state.get("theme", "Light")
    theme_vals = _get_theme_values(theme_choice)
    inject_css(theme_vals)

    st.markdown(f"""
    <div class="hero">
        <h1>🛍️ Smart Product Scanner</h1>
        <p class="muted">Upload an image or take a photo to instantly find visually similar products</p>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()
    if not os.path.exists(METADATA_FILE):
        generate_mock_dataset()
    if not os.path.exists(EMBEDDINGS_FILE):
        generate_embeddings(model)

    all_features = np.load(EMBEDDINGS_FILE)
    with open(METADATA_FILE, "r") as f:
        product_metadata = json.load(f)

    input_image = uploaded_file or camera_input

    if input_image is not None:
        st.markdown('<div class="query-card"><h3>📌 Your Query Image</h3>', unsafe_allow_html=True)
        query_img = Image.open(input_image)
        st.image(query_img, width=320)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        with st.spinner("🔍 Analyzing image and finding similar products..."):
            input_features = extract_features(model, input_image)
            similar_products = find_similar_products(input_features, all_features, product_metadata)

            st.subheader("✨ Top Matches")
            cols = st.columns(NUM_SIMILAR_PRODUCTS)
            for i, (product, score) in enumerate(similar_products[:NUM_SIMILAR_PRODUCTS]):
                with cols[i]:
                    img_b64 = image_to_base64(product["image_path"])
                    st.markdown(
                        f"""
                        <div class="product-card">
                            <img src="data:image/jpeg;base64,{img_b64}" class="product-image" />
                            <div class="product-title">{product['name']}</div>
                            <div class="product-price">${product['price']:.2f}</div>
                            <progress value="{score:.4f}" max="1" style="width:100%"></progress>
                            <small class="muted">Similarity: {score:.2%}</small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
    else:
        st.info("👆 Upload an image or use the camera to begin your search.")
        st.markdown("<p style='text-align:center' class='muted'>Your perfect product match is one photo away.</p>",
                    unsafe_allow_html=True)


if __name__ == "__main__":
    main()
