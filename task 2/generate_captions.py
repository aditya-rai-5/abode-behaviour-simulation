"""
Lightweight script to generate image captions in batches using the Qwen2-VL model.

This is a runnable adaptation of the logic in `image_captioning.ipynb`.

Usage (PowerShell):
python "task 2\generate_captions.py" --input_csv "path/to/test_dataset.csv" --output_csv "path/to/generated_captions.csv" --batch_size 32

Notes:
- This script assumes you have the required packages installed (see ../requirements.txt).
- Model & processor names may need to be adjusted depending on availability and access tokens.
- For large datasets, run in Colab or a machine with a GPU.
"""

import argparse
import math
import time
import pandas as pd
from tqdm import tqdm

# The heavy imports are wrapped to provide friendlier error messages if packages are missing
try:
    import torch
    from transformers import AutoProcessor, AutoModelForConditionalGeneration
    from PIL import Image
    import requests
    from io import BytesIO
except Exception as e:
    raise ImportError("Missing dependencies. Please install packages from requirements.txt (torch, transformers, pillow, requests).\nOriginal error: {}".format(e))


def load_model(model_path="Qwen/Qwen2-VL-2B-Instruct", quantize=False):
    """Load the processor and model. Quantization options are intentionally minimal here.

    Adjust `model_path` to match the model you want to use. If you have a local copy
    or require authentication, set the appropriate env vars or token in the notebook.
    """
    print(f"Loading model: {model_path}...")

    # Use AutoModelForConditionalGeneration as a generic fallback name; if you used a custom class in the notebook,
    # replace this with the appropriate class (e.g., Qwen2VLForConditionalGeneration) when available.
    model = AutoModelForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    processor = AutoProcessor.from_pretrained(model_path)

    print("Model and processor loaded.")
    return model, processor


def load_image(image_path_or_url, max_size=(512, 512)):
    try:
        if str(image_path_or_url).startswith("http"):
            response = requests.get(image_path_or_url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            img = Image.open(image_path_or_url).convert("RGB")

        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        # Return None on failure so caller can mark the row as failed
        return None


def caption_images_batch(model, processor, image_files_list, prompt, max_new_tokens=256):
    """Generate captions for a batch of image files (urls or local paths).

    Returns a list of strings (captions) aligned with image_files_list. If an image failed
    to load, the returned caption for that position is an error message.
    """
    device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"

    images = []
    valid_indices = []
    for i, image_file in enumerate(image_files_list):
        img = load_image(image_file)
        if img is None:
            images.append(None)
        else:
            images.append(img)
            valid_indices.append(i)

    if not any(images):
        return ["Error: No valid images in batch"] * len(image_files_list)

    # Prepare the conversation-style prompts for each valid image (processor handles chat templates)
    conversations = []
    for _ in range(len([img for img in images if img is not None])):
        conversations.append([
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ])

    # Build text prompts using the processor (similar to notebook behaviour)
    text_prompts = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)

    # Prepare inputs; pair images and text prompts using the processor
    valid_images = [img for img in images if img is not None]
    inputs = processor(text=text_prompts, images=valid_images, return_tensors="pt", padding=True).to(device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # Remove the input portion from outputs
    input_token_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_token_len:]

    responses = processor.batch_decode(generated_ids, skip_special_tokens=True)
    cleaned = [r.strip() for r in responses]

    # Map back into original image_files_list order
    final_output = []
    resp_idx = 0
    for img in images:
        if img is None:
            final_output.append("Error: Image load failed")
        else:
            final_output.append(cleaned[resp_idx])
            resp_idx += 1

    return final_output


def main(input_csv, output_csv, model_path, prompt_text, batch_size=32, max_rows=None):
    print(f"Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    if max_rows:
        df = df.iloc[:max_rows]

    # Ensure the dataset has a column named `media_url` (the notebook uses this). If not, try `media`.
    media_col = None
    for candidate in ["media_url", "media", "media_url"]:
        if candidate in df.columns:
            media_col = candidate
            break

    if media_col is None:
        raise ValueError("No media column found in input CSV. Expected 'media_url' or 'media'.")

    # Prepare output column
    df["generated_caption"] = ""

    model, processor = load_model(model_path)

    total_batches = math.ceil(len(df) / batch_size)
    start_time_all = time.time()

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i : i + batch_size]
        image_list = batch_df[media_col].astype(str).tolist()

        print(f"Processing batch {i//batch_size + 1}/{total_batches} (size {len(image_list)})...")
        captions = caption_images_batch(model, processor, image_list, prompt_text)

        for j, cap in enumerate(captions):
            df.at[batch_df.index[j], "generated_caption"] = cap

        # Save intermediate results for resilience
        df.to_csv(output_csv, index=False)

    end_time_all = time.time()
    print(f"Done. Total time: {(end_time_all - start_time_all)/60:.2f} minutes.")
    print(f"Final results saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions for images in a CSV file.")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV containing a media_url column")
    parser.add_argument("--output_csv", required=True, help="Path to write output CSV with generated_caption column")
    parser.add_argument("--model_path", default="Qwen/Qwen2-VL-2B-Instruct", help="Hugging Face model path or local path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for image processing")
    parser.add_argument("--max_rows", type=int, default=None, help="(Optional) limit number of rows to process")
    parser.add_argument("--prompt", type=str, default=(
        "Describe everything visible in the image with maximum objective detail, "
        "including objects, environment, people, text, vehicle features, materials, lighting, spatial layout, textures, and reflections."
    ), help="Shared text prompt for the captioning model")

    args = parser.parse_args()

    main(args.input_csv, args.output_csv, args.model_path, args.prompt, batch_size=args.batch_size, max_rows=args.max_rows)
