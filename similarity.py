# similarity.py
# Description: Contains the function to find similar products using cosine similarity.

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_products(input_features, all_features, product_metadata):
    """
    Finds similar products by calculating cosine similarity between feature vectors.
    """
    # Reshape input features to be a 2D array for the similarity function
    input_features_reshaped = input_features.reshape(1, -1)

    # Calculate cosine similarities
    similarities = cosine_similarity(input_features_reshaped, all_features)

    # Get indices of the most similar products, sorted from highest to lowest similarity
    similar_indices = np.argsort(similarities.flatten())[::-1]

    # Prepare results list
    results = []
    # --- START: CORRECTION ---
    # The filter to remove the exact match has been removed.
    # We now iterate through all sorted indices to include the identical item if found.
    for i in similar_indices:
        results.append((product_metadata[i], similarities[0, i]))
    # --- END: CORRECTION ---

    return results