import requests
from PIL import Image
from io import BytesIO
import time
import pandas as pd
import math
import os
import base64
from concurrent.futures import ThreadPoolExecutor

# --- 1. Configuration ---

# API key from your uploaded 'caption.py'
apiKey = "AIzaSyDaF-EeOyXKQ_zI-1ONCcqnAVgSx99C2Jo" 

# Prompt from your 'caption.py'
text_prompt = "Generate a concise, caption included objects, text, environment and information for this image. give in one para with details"

# --- Script Settings ---
# This must be the *original* data file
ORIGINAL_CSV_FILE = 'train_dataset.csv' 
# This is the log file from the 'download_images.py' script
LOG_CSV_FILE = 'download_log.csv' 
OUTPUT_CSV_FILE = 'final_captions_output.csv'

# Settings for speed
MAX_WORKERS = 50 # Number of parallel API calls.
SAVE_INTERVAL = 100 # Save progress every 500 images

# --- 2. Helper Functions ---

def _load_and_resize_image(local_image_path, max_size=(1024, 1024)):
    """
    Loads, resizes, and base64-encodes a single LOCAL image.
    Returns (base64_string, mime_type) or (None, error_message)
    """
    try:
        # Load from local disk
        image = Image.open(local_image_path).convert('RGB')
        
        # Resize for faster upload
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert to JPEG for smaller file size
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return img_base64, "image/jpeg"
        
    except Exception as e:
        print(f"ERROR [Image Load]: {local_image_path} - {e}")
        return None, f"Error: Image load/resize failed ({e})"

def _call_gemini_api(img_base64, mime_type, prompt):
    """
    Calls the Gemini API with exponential backoff.
    Returns (caption_text) or (error_message)
    """
    
    # --- MODIFIED LINE ---
    # Updated the model name to gemini-2.5-flash-image
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-live:generateContent?key={apiKey}"    # --- END MODIFICATION ---
    
    headers = {'Content-Type': 'application/json'}
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": img_base64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 200 # From your 'caption.py'
        }
    }
    
    # Implement exponential backoff for API rate limits
    max_retries = 3
    delay = 1
    for attempt in range(max_retries):
        try:
            response = requests.post(apiUrl, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 429:
                print(f"WARN: Rate limit hit. Retrying in {delay}s...")
                time.sleep(delay + (attempt * 2)) # Increase delay
                delay *= 2 
                continue

            response.raise_for_status() # Check for other HTTP errors (4xx, 5xx)
            
            result = response.json()
            
            if 'candidates' in result and result['candidates'][0].get('content', {}).get('parts', []):
                caption = result['candidates'][0]['content']['parts'][0]['text']
                return caption.strip()
            elif 'error' in result:
                return f"Error: API Error ({result['error']['message']})"
            else:
                if 'promptFeedback' in result:
                    block_reason = result['promptFeedback'].get('blockReason', 'Unknown')
                    print(f"WARN: Prompt blocked ({block_reason})")
                    return f"Error: Prompt blocked ({block_reason})"
                return "Error: Unknown API response"
                
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return f"Error: API request failed ({e})"
            time.sleep(delay)
            delay *= 2
            
    return "Error: Max retries exceeded"

def _process_row(task_args):
    """
    Main function for each thread. Takes one task, processes it, and returns the result.
    A task is a tuple: (original_index, local_image_path)
    """
    original_index, local_image_path = task_args
    
    # 1. Load, resize, and base64-encode the local image
    img_base64, result = _load_and_resize_image(local_image_path)
    if img_base64 is None:
        return original_index, result # 'result' contains the error message
        
    # 2. Call the API to get the caption
    caption = _call_gemini_api(img_base64, result, text_prompt) # 'result' contains mime_type
    
    return original_index, caption

# --- 3. Main Batch Processing Script ---

def main_processing():
    
    if not apiKey:
        print("Error: 'apiKey' variable is empty.")
        print("Please get a key from https://aistudio.google.com/app/apikey and paste it into the script.")
        return

    # Check for required files
    if not os.path.exists(ORIGINAL_CSV_FILE):
        print(f"Error: Original data file not found at {ORIGINAL_CSV_FILE}")
        return
    if not os.path.exists(LOG_CSV_FILE):
        print(f"Error: Download log file not found at {LOG_CSV_FILE}")
        print("Please run the 'download_images.py' script first.")
        return

    # Load original dataset
    print(f"Loading original dataset from {ORIGINAL_CSV_FILE}...")
    df = pd.read_csv(ORIGINAL_CSV_FILE)
    
    # Load the download log
    print(f"Loading download log from {LOG_CSV_FILE}...")
    log_df = pd.read_csv(LOG_CSV_FILE)
    
    if 'generated_caption' not in df.columns:
        df['generated_caption'] = None
        
    print(f"Dataset loaded. Total rows: {len(df)}")

    # --- Prepare tasks ---
    tasks_to_run = []
    
    # Use the log file to find which rows to process
    for _, log_row in log_df.iterrows():
        original_index = log_row['original_index']
        status_or_path = log_row['status_or_path']
        
        # Skip if a caption already exists (from a previous run)
        if pd.notna(df.at[original_index, 'generated_caption']):
            continue
            
        # Check if the log entry is a valid file path
        if status_or_path and status_or_path.endswith('.jpg'):
            # This is a valid, downloaded image. Add to task list.
            tasks_to_run.append((original_index, status_or_path))
        else:
            # This is an error or "No Media Found". Copy the status.
            df.at[original_index, 'generated_caption'] = status_or_path

    if not tasks_to_run:
        print("Processing complete. No new images to caption.")
        df.to_csv(OUTPUT_CSV_FILE, index=False)
        return

    total_tasks = len(tasks_to_run)
    print(f"Found {total_tasks} new images to caption.")
    
    total_start_time = time.time()
    print(f"Starting parallel processing with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # executor.map runs _process_row for every item in 'tasks_to_run'
        results = executor.map(_process_row, tasks_to_run)
        
        # Process results as they come in
        for i, (index, caption) in enumerate(results):
            df.at[index, 'generated_caption'] = caption
            
            # Print progress and save checkpoint
            if (i + 1) % SAVE_INTERVAL == 0:
                elapsed = time.time() - total_start_time
                avg_time = elapsed / (i + 1)
                remaining = (total_tasks - (i + 1)) * avg_time
                
                print(f"--- Processed {i+1}/{total_tasks} ---")
                print(f"--- Saving intermediate results to {OUTPUT_CSV_FILE} ---")
                print(f"--- Est. time remaining: {remaining/60:.2f} minutes ---")
                df.to_csv(OUTPUT_CSV_FILE, index=False)

    # --- 4. Final Save ---
    total_end_time = time.time()
    print("\n--- Processing Complete ---")
    print(f"Total time taken: {(total_end_time - total_start_time) / 60:.2f} minutes.")
    print(f"Saving final results to {OUTPUT_CSV_FILE}...")
    df.to_csv(OUTPUT_CSV_FILE, index=False)
    print("Done.")

# This makes the script runnable
if __name__ == "__main__":
    main_processing()