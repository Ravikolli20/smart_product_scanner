# scraper.py
# Description: A modular web scraper to build the product database
#              from multiple e-commerce websites (Amazon, Snapdeal).

import os
import json
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import random

# Import project configuration
from config import IMAGE_DIR, METADATA_FILE

# --- Rotatinhg User-Agents to avoid getting blocked ---
USER_AGENT_LIST = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
]

def get_random_headers():
    """Returns a dictionary of headers with a random User-Agent."""
    return {
        "User-Agent": random.choice(USER_AGENT_LIST),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    }

def scrape_from_amazon(query: str, category_name: str, product_id_counter: int):
    """Scrapes a single category from Amazon.in."""
    print(f"\nScraping '{category_name}' from Amazon...")
    base_url = "https://www.amazon.in"
    search_url = f"{base_url}/s?k={query.replace(' ', '+')}"
    products = []

    try:
        time.sleep(random.uniform(2, 4))
        response = requests.get(search_url, headers=get_random_headers())
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        product_pods = soup.find_all('div', {'data-component-type': 's-search-result'})
        
        if not product_pods:
            print(f"⚠️ Amazon: No products found for '{query}'. May be blocked.")
            return [], product_id_counter

        for pod in tqdm(product_pods, desc=f"Processing {category_name}"):
            name_element = pod.find('h2', class_='a-size-mini')
            if not name_element: continue
            
            price_element = pod.find('span', class_='a-price-whole')
            if not price_element: continue

            link_element = pod.find('a', class_='a-link-normal', href=True)
            if not link_element: continue

            image_element = pod.find('img', class_='s-image')
            if not image_element or not image_element.has_attr('src'): continue

            image_path = os.path.join(IMAGE_DIR, f"product_{product_id_counter:04d}.jpg")
            img_response = requests.get(image_element['src'], stream=True, headers=get_random_headers())
            if img_response.status_code == 200:
                with open(image_path, 'wb') as f:
                    f.write(img_response.content)
            else:
                continue

            products.append({
                "id": f"product_{product_id_counter:04d}",
                "name": name_element.text.strip(),
                "category": category_name,
                "price": float(price_element.text.replace(',', '').strip()),
                "product_url": base_url + link_element['href'],
                "image_path": image_path.replace("\\", "/")
            })
            product_id_counter += 1
            time.sleep(random.uniform(0.5, 1.5))
            
    except Exception as e:
        print(f"❌ Amazon: An error occurred for query '{query}': {e}")
    
    return products, product_id_counter

def scrape_from_snapdeal(query: str, category_name: str, product_id_counter: int):
    """Scrapes a single category from Snapdeal.com."""
    print(f"\nScraping '{category_name}' from Snapdeal...")
    base_url = "https://www.snapdeal.com"
    search_url = f"{base_url}/search?keyword={query.replace(' ', '%20')}"
    products = []

    try:
        time.sleep(random.uniform(2, 4))
        response = requests.get(search_url, headers=get_random_headers())
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        product_pods = soup.find_all('div', class_='product-tuple-listing')

        if not product_pods:
            print(f"⚠️ Snapdeal: No products found for '{query}'. May be blocked or selectors are outdated.")
            return [], product_id_counter

        for pod in tqdm(product_pods, desc=f"Processing {category_name}"):
            try:
                name_element = pod.find('p', class_='product-title')
                if not name_element: continue

                price_element = pod.find('span', class_='product-price')
                if not price_element: continue
                price_str = price_element.get('data-price', '0').strip()

                link_element = pod.find('a', class_='dp-widget-link', href=True)
                if not link_element: continue

                image_element = pod.find('img', class_='product-image')
                if not image_element or not (image_element.has_attr('src') or image_element.has_attr('data-src')): continue
                image_url = image_element.get('src') or image_element.get('data-src')

                image_path = os.path.join(IMAGE_DIR, f"product_{product_id_counter:04d}.jpg")
                img_response = requests.get(image_url, stream=True, headers=get_random_headers())
                if img_response.status_code == 200:
                    with open(image_path, 'wb') as f:
                        f.write(img_response.content)
                else:
                    continue

                products.append({
                    "id": f"product_{product_id_counter:04d}",
                    "name": name_element.text.strip(),
                    "category": category_name,
                    "price": float(price_str),
                    "product_url": link_element['href'],
                    "image_path": image_path.replace("\\", "/")
                })
                product_id_counter += 1
                time.sleep(random.uniform(0.5, 1.5))
            
            except Exception:
                continue
            
    except Exception as e:
        print(f"❌ Snapdeal: An error occurred for query '{query}': {e}")
        
    return products, product_id_counter


def run_scraper():
    """
    Main function to orchestrate the scraping process from multiple sites.
    """
    print("🚀 Starting multi-site scraping process...")
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # --- START: CORRECTION ---
    # Updated task list to use Snapdeal for most categories
    tasks = [
        {'query': "mens watches", 'category': "Watches", 'site': "amazon"},
        {'query': "running shoes for men", 'category': "Shoes", 'site': "amazon"},
        {'query': "laptops", 'category': "Laptops", 'site': "snapdeal"},
        {'query': "smartphones", 'category': "Mobiles", 'site': "snapdeal"},
        {'query': "office chairs", 'category': "Furniture", 'site': "snapdeal"},
        {'query': "tablets", 'category': "Tablets", 'site': "snapdeal"},
        {'query': "bluetooth headphones", 'category': "Headphones", 'site': "snapdeal"},
        {'query': "water bottles", 'category': "Accessories", 'site': "snapdeal"},
    ]
    
    # Updated dispatcher to remove Ajio
    scraper_dispatcher = {
        "amazon": scrape_from_amazon,
        "snapdeal": scrape_from_snapdeal,
    }
    # --- END: CORRECTION ---

    all_products = []
    product_id_counter = 0

    for task in tasks:
        scraper_func = scraper_dispatcher.get(task['site'])
        if scraper_func:
            new_products, new_counter = scraper_func(task['query'], task['category'], product_id_counter)
            all_products.extend(new_products)
            product_id_counter = new_counter
        else:
            print(f"⚠️ No scraper found for site: {task['site']}")

    # --- Save all collected data to a single JSON file ---
    if all_products:
        with open(METADATA_FILE, 'w') as f:
            json.dump(all_products, f, indent=4)
        print(f"\n✅ Success! Scraped a total of {len(all_products)} products from multiple sites.")
        print(f"💾 Database saved to {METADATA_FILE}")
        print(f"🖼️ Images saved in '{IMAGE_DIR}'")
    else:
        print("\n❌ Failed to scrape any products. Please check the script and your connection.")

if __name__ == '__main__':
    run_scraper()
