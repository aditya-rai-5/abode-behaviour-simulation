import requests
from PIL import Image
from io import BytesIO
import time
import pandas as pd
import math
import os
from concurrent.futures import ThreadPoolExecutor

# --- 1. Configuration ---
INPUT_CSV_FILE = 'content_simulation_train.csv' 
LOG_CSV_FILE = 'download_log.csv'
DOWNLOAD_FOLDER = 'image_downloads'
MEDIA_COLUMN_NAME = 'media'
ID_COLUMN_NAME = 'id' # Column to use for unique filenames
NO_MEDIA_STRING = "no media found"
MAX_WORKERS = 50 # Number of parallel downloads. 50 is a good start.
SAVE_INTERVAL = 500 # Save progress every 500 images

# --- 2. Helper Functions ---

def thumb_extract(url):
    """
    Extracts the actual image URL from the media string.
    """
    try:
        return str(url).split("'")[1]
    except Exception:
        return None

def download_and_save_image(args):
    """
    Downloads, resizes, and saves a single image.
    This function is run in parallel by the thread pool.
    """
    index, image_url, save_path = args
    max_size = (1024, 1024)
    
    try:
        # 1. Download image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status() # Check for download errors
        image = Image.open(BytesIO(response.content)).convert('RGB')
        
        # 2. Resize image
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # 3. Save image as JPEG
        image.save(save_path, format="JPEG")
        
        # Return the original index and the new local path
        return index, save_path
        
    except Exception as e:
        print(f"ERROR [Download Failed]: {image_url} - {e}")
        # Return the original index and an error message
        return index, f"Error: Download failed ({e})"

# --- 3. Main Download Script ---

def main_download():
    # Check if input file exists
    if not os.path.exists(INPUT_CSV_FILE):
        print(f"Error: Input file not found at {INPUT_CSV_FILE}")
        print("Please update the INPUT_CSV_FILE variable in the script.")
        return

    # Create the download folder if it doesn't exist
    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)
        print(f"Created folder: {DOWNLOAD_FOLDER}")

    # Load dataset
    print(f"Loading dataset from {INPUT_CSV_FILE}...")
    df = pd.read_csv(INPUT_CSV_FILE)
    print(f"Dataset loaded. Total rows: {len(df)}")

    # Prepare the log DataFrame
    # If a log file already exists, load it to resume progress
    if os.path.exists(LOG_CSV_FILE):
        print(f"Resuming from existing log file: {LOG_CSV_FILE}")
        log_df = pd.read_csv(LOG_CSV_FILE)
    else:
        print("Creating new log file...")
        # Create a new DataFrame to log results
        log_df = pd.DataFrame(columns=['original_index', 'status_or_path'])

    # Convert log_df to a set for fast lookup of processed items
    processed_indices = set(log_df['original_index'])
    print(f"Found {len(processed_indices)} already processed rows.")

    # Create a list of tasks to run
    tasks = []
    for index, row in df.iterrows():
        # Skip if this row's index is already in the log file
        if index in processed_indices:
            continue
            
        media_data = str(row[MEDIA_COLUMN_NAME])
        
        if NO_MEDIA_STRING in media_data.lower() or pd.isna(media_data):
            tasks.append((index, "No Media Found"))
        else:
            image_url = thumb_extract(media_data)
            if image_url:
                # Create a unique filename based on the ID column
                # This prevents errors if IDs are not unique numbers
                file_id = str(row[ID_COLUMN_NAME]).replace(" ", "_").replace("/", "_")
                save_path = os.path.join(DOWNLOAD_FOLDER, f"id_{file_id}.jpg")
                tasks.append((index, image_url, save_path))
            else:
                tasks.append((index, "Error: URL extraction failed"))

    if not tasks:
        print("No new rows to process. Exiting.")
        return

    print(f"Found {len(tasks)} new rows to download/process.")
    total_tasks = len(tasks)
    total_start_time = time.time()
    
    # We will append results to this list
    new_log_entries = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Separate tasks that need downloading from tasks that are just logging errors
        download_tasks = []
        log_only_tasks = []
        
        for task_data in tasks:
            if len(task_data) == 3: # (index, url, save_path)
                download_tasks.append(task_data)
            else: # (index, error_message)
                log_only_tasks.append(task_data)

        # 1. Add all the log-only tasks to the results immediately
        new_log_entries.extend(log_only_tasks)
        
        print(f"Starting parallel download of {len(download_tasks)} images...")
        
        # 2. Run all download tasks in parallel
        # We use executor.map for the download tasks
        results = executor.map(download_and_save_image, download_tasks)
        
        # 3. Process results as they come in
        for i, (index, status_or_path) in enumerate(results):
            new_log_entries.append((index, status_or_path))
            
            # Print progress and save checkpoint
            current_processed = i + 1 + len(log_only_tasks)
            if current_processed % SAVE_INTERVAL == 0:
                elapsed = time.time() - total_start_time
                avg_time = elapsed / current_processed
                remaining = (total_tasks - current_processed) * avg_time
                
                print(f"--- Processed {current_processed}/{total_tasks} ---")
                print(f"--- Saving intermediate log to {LOG_CSV_FILE} ---")
                
                # Append new entries to the log DataFrame and save
                temp_df = pd.DataFrame(new_log_entries, columns=['original_index', 'status_or_path'])
                log_df = pd.concat([log_df, temp_df], ignore_index=True)
                log_df.to_csv(LOG_CSV_FILE, index=False)
                new_log_entries = [] # Clear the temporary list

    # --- 4. Final Save ---
    # Save any remaining entries
    if new_log_entries:
        temp_df = pd.DataFrame(new_log_entries, columns=['original_index', 'status_or_path'])
        log_df = pd.concat([log_df, temp_df], ignore_index=True)
    
    total_end_time = time.time()
    print("\n--- Download Complete ---")
    print(f"Total time taken: {(total_end_time - total_start_time) / 60:.2f} minutes.")
    print(f"Saving final log file to {LOG_CSV_FILE}...")
    log_df.to_csv(LOG_CSV_FILE, index=False)
    print("Done.")

# This makes the script runnable
if __name__ == "__main__":
    main_download()
