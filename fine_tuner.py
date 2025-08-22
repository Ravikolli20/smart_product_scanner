# fine_tuner.py
# Description: A script to fine-tune the MobileNetV2 model on our scraped product dataset.

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import tqdm

# Import project modules
from config import METADATA_FILE, IMG_SIZE, IMAGE_DIR

# --- Configuration ---
BATCH_SIZE = 32
EPOCHS = 10 # Number of times to train on the full dataset
MODEL_SAVE_PATH = 'fine_tuned_model.keras'

def prepare_dataset():
    """
    Loads the scraped product data, validates images using TensorFlow,
    and prepares training and validation datasets.
    """
    print("Preparing dataset for fine-tuning...")
    with open(METADATA_FILE, 'r') as f:
        products = json.load(f)

    print("Validating images using TensorFlow...")
    valid_image_paths = []
    valid_labels = []
    for product in tqdm(products, desc="Checking images"):
        path = product.get('image_path')
        if path and os.path.exists(path):
            try:
                # --- START: CORRECTION ---
                # Use TensorFlow's own functions to validate the image.
                # This is the most reliable way to prevent training errors.
                img_bytes = tf.io.read_file(path)
                tf.image.decode_image(img_bytes, channels=3)
                # --- END: CORRECTION ---
                valid_image_paths.append(path)
                valid_labels.append(product['category'])
            except tf.errors.InvalidArgumentError:
                # This will catch the exact error that was causing the crash.
                # print(f"Skipping corrupted image: {path}")
                pass
        
    if not valid_image_paths:
        print("❌ Error: No valid images found. Please run the scraper first.")
        return None, None, None

    # --- Encode Labels ---
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(valid_labels)
    num_classes = len(label_encoder.classes_)
    print(f"Found {len(valid_image_paths)} valid images across {num_classes} categories.")

    # --- Split Data ---
    X_train, X_val, y_train, y_val = train_test_split(
        valid_image_paths, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )

    # --- Create TensorFlow Datasets ---
    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE[0], IMG_SIZE[1]])
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, label

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=100).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, val_dataset, num_classes

def fine_tune_model(train_ds, val_ds, num_classes):
    """
    Loads the pre-trained MobileNetV2 model, freezes its base layers,
    adds new classification layers, and fine-tunes it on our data.
    """
    print("Starting the fine-tuning process...")
    
    # --- Load Base Model ---
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )

    # --- Freeze Base Layers ---
    base_model.trainable = False

    # --- Add Custom Layers ---
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x) # Name the layer
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)

    # --- Compile and Train ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Training the model...")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds
    )

    # --- Save the Fine-Tuned Model ---
    model.save(MODEL_SAVE_PATH)
    print(f"✅ Model fine-tuning complete! Saved to '{MODEL_SAVE_PATH}'")

if __name__ == '__main__':
    train_dataset, val_dataset, num_classes = prepare_dataset()
    if train_dataset:
        fine_tune_model(train_dataset, val_dataset, num_classes)
