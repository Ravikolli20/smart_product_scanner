🛍️ Smart Product Scanner
A sophisticated web application that allows users to find visually similar products across multiple e-commerce websites. Users can upload an image or take a photo of a product, and the app's machine learning backend will return a list of the most similar items, complete with prices and direct links to the product pages.

✨ Features
Visual Search Engine: Uses a deep learning model (MobileNetV2) to analyze images and find products based on visual features, not just keywords.

Automated Multi-Site Database: A powerful web scraper automatically builds a rich product database by scraping data from major e-commerce sites like Amazon and Snapdeal.

Clickable, Shoppable Results: Product matches are displayed in beautiful, interactive cards that link directly to the e-commerce page for purchase.

Dual Input Methods: Supports both file uploads and direct camera access for a seamless user experience on desktop and mobile.

Elegant & Intuitive UI: A modern, responsive user interface built with Streamlit, featuring a custom theme with a Light/Dark mode toggle.

🔧 Tech Stack
Backend & ML: Python, TensorFlow/Keras

Web Framework: Streamlit

Web Scraping: BeautifulSoup, Requests

Data Handling: NumPy, Pillow

Core Model: MobileNetV2 (pre-trained on ImageNet)

Similarity Algorithm: Cosine Similarity (via Scikit-learn)

🚀 Setup and Installation
Follow these steps to get the project running on your local machine.

1. Clone the Repository
git clone [https://github.com/your-username/smart-product-scanner.git](https://github.com/your-username/smart-product-scanner.git)
cd smart-product-scanner

2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

Using conda:

conda create --name smartproduct python=3.9
conda activate smartproduct

Using venv:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install Dependencies
All required packages are listed in the requirements.txt file.

pip install -r requirements.txt

4. Build the Product Database
The first time you run the project, you need to build the database using the web scraper. This script will create the product_data directory, scrape e-commerce sites, download images, and create the products.json file.

python scraper.py

Note: This step can take a few minutes as it scrapes multiple websites.

5. Run the Streamlit App
Once the scraper has successfully built the database, you can launch the main application.

streamlit run app.py

The application should now be open and running in your web browser!

🛠️ How It Works
The project is divided into two main components: the Web Scraper and the Streamlit Application.

Web Scraper (scraper.py):

This script is run once to build the product database.

It visits predefined e-commerce websites (Amazon, Snapdeal).

It extracts product information (name, price, image, URL) and saves it to products.json.

It downloads the product images into the product_data/product_images directory.

Streamlit Application (app.py):

Data Preprocessing: On its first run after the database is built, the app processes every product image.

Feature Extraction: The pre-trained MobileNetV2 model converts each image into a high-dimensional feature vector (embedding). These embeddings are saved in embeddings.npy for fast access.

User Input: The user uploads an image or takes a photo.

Similarity Matching: The input image is converted into a feature vector using the same model. This vector is then compared against all the vectors in the embeddings.npy database using Cosine Similarity.

Display Results: The top N most similar products are retrieved and displayed in clickable, styled cards in the UI.

💡 Future Enhancements
Expand Scraper: Add more e-commerce sites (e.g., Ajio, Myntra) and improve error handling to make the scraper even more robust.

Fine-Tune the Model: For higher accuracy, the MobileNetV2 model could be fine-tuned on a larger, custom dataset of product images.

Admin Interface: Build an "Add Product" page directly into the app to allow for easy manual additions to the database without running the scraper.

User Accounts & Favorites: Add user authentication to allow users to save their favorite items.
