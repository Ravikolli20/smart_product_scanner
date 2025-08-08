# config.py
# Description: Contains all configurations, constants, and file paths for the application.

import os

# --- DIRECTORY AND FILE PATHS ---

# Base directory for all data
DATA_DIR = "product_data"

# Subdirectory for product images
IMAGE_DIR = os.path.join(DATA_DIR, "product_images")

# Path to the JSON file containing product metadata (name, price, etc.)
METADATA_FILE = os.path.join(DATA_DIR, "products.json")

# Path to the .npy file storing the pre-computed feature embeddings
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")


# --- MODEL AND IMAGE CONFIGURATION ---

# The target size for images to be fed into the neural network
IMG_SIZE = (224, 224)


# --- APPLICATION SETTINGS ---

# The number of similar products to display in the results
NUM_SIMILAR_PRODUCTS = 4