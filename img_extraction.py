import pandas as pd
import os
import requests
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm.notebook import tqdm

data=pd.read_csv("train_dataset.csv")
# Let's set a standard, high-performance size
# 224x224 is a very common and effective size for ViT models
TARGET_IMG_SIZE = (224, 224) 
img_dir = "images/"
os.makedirs(img_dir, exist_ok=True)

# Get unique URLs to avoid re-downloading
unique_urls = data['media_url'].unique()
url_to_path_map = {}

def download_resize_and_save(url):
    if url == "no_media" or pd.isna(url):
        return url, None
    
    try:
        # Create a simple, safe filename
        filename = os.path.join(img_dir, str(hash(url)) + ".jpg")
        
        # Only download if it doesn't already exist
        if not os.path.exists(filename):
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            
            with Image.open(BytesIO(resp.content)) as img:
                # --- THIS IS THE NEW STEP ---
                # Convert to RGB, resize, and then save
                img = img.convert("RGB").resize(TARGET_IMG_SIZE, Image.Resampling.LANCZOS)
                # --------------------------
                
                img.save(filename, "JPEG")
                
        return url, filename
    except Exception as e:
        # If download fails, we'll map it to None
        return url, None

with ThreadPoolExecutor(max_workers=32) as executor:
    results = list(tqdm(executor.map(download_resize_and_save, unique_urls), total=len(unique_urls), desc="Downloading images"))

for url, path in results:
    url_to_path_map[url] = path

data['local_image_path'] = data['media_url'].map(url_to_path_map)

failed_downloads = data['local_image_path'].isna().sum() - (data['media_url'] == 'no_media').sum()
print(f"Image pre-downloading complete.")
print(f"Total failed/missing media (will use placeholder): {failed_downloads}")

data.to_csv("train_dataset.csv", index=False)
print("Updated train_dataset.csv with local_image_path")
