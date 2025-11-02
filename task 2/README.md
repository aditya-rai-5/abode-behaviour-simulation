# Abode Behaviour Simulation

This project implements an advanced social media marketing content generation system using Large Language Models (LLMs) and computer vision techniques. It simulates marketing behavior by generating contextually appropriate content based on various inputs including images, metadata, and brand identity.

## Project Structure

# Task 2 — Image Captioning & Content Generation (Fine-tune / Inference)

This folder contains notebooks and scripts used for Task 2: generating descriptive image captions, building prompts, and fine-tuning a LLaMA-based content-generation adapter to simulate marketing tweets.

What this folder contains
------------------------
- `data_cleaning.py` — lightweight preprocessing: date parsing, media URL extraction, media type tagging, column renaming and small sanity checks. Update the input/output filenames at the top of the script before running.
- `image_captioning.ipynb` — a notebook that loads a Qwen2-VL model (quantized) and generates detailed captions for images (URLs or local files). It supports batching, intermediate saves, and thumbnailing for faster processing.
- `fine_tune_llama.ipynb` — the fine-tuning notebook. It:
  - prepares prompts using `prompt.py` logic
  - applies LLaMA chat templates
  - configures LoRA and 4-bit quantization
  - creates a formatted dataset for SFTTrainer and trains/saves an adapter
- `prompt.py` — prompt-building function(s). Used during data prep and to create inference prompts.
- `metric.py` — evaluation script that computes BLEU, ROUGE and CIDEr on a CSV with columns `content` (reference) and `generated_content` (candidate). Update `RESULTS_CSV_PATH` before running.
- `test pipeline.ipynb` — an end-to-end inference notebook that:
  - loads a fine-tuned adapter + base model
  - uses `image_captioning` outputs (generated captions) as inputs
  - creates the final input prompts and runs batched generation
  - saves `generated_content` to a CSV for downstream evaluation

Data expectations
-----------------
Input tabular files should contain (or be adapted to provide) these fields:

- `date` — timestamp of the post (parseable by pandas)
- `likes` — historical likes count (training only)
- `username` — handle (used for brand identity)
- `inferred company` or `inferred_company` — inferred brand name
- `media` / `media_url` — image URL or descriptor; `data_cleaning.py` extracts the usable URL into `media_url`
- `content` — original tweet text (for training/evaluation)

Quick start (Colab)
-------------------
1. Open `image_captioning.ipynb` or `fine_tune_llama.ipynb` in Colab.
2. Mount Google Drive (cells include mount commands).
3. Install required libs (cells include pip install commands for `bitsandbytes`, `transformers`, `trl`, etc.).
4. Update paths (e.g., `/content/test_dataset.csv`, Drive save paths) and run cells sequentially.

Quick start (Local / Windows PowerShell)
--------------------------------------
1. Create and activate a venv:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install pandas numpy torch transformers bitsandbytes accelerate peft trl datasets pillow rouge-score pycocoevalcap nltk
```

2. Prepare your dataset and set correct file paths in `data_cleaning.py` and notebooks.
3. Run `data_cleaning.py` to create `media_url` and `media_type` columns:

```powershell
python "task 2\data_cleaning.py"
```

4. Run `image_captioning.ipynb` (or extract relevant cells to a script) to produce `generated_caption` for each media.

5. Use `fine_tune_llama.ipynb` to build prompts (scripted in the notebook using `prompt.py`) and fine-tune the adapter. Adjust `model_name`, `bnb_config`, LoRA params and dataset path as needed.

6. Run `test pipeline.ipynb` to load the adapter and generate `generated_content` for evaluation.

Evaluation
----------
- Use `metric.py` to compute BLEU, ROUGE and CIDEr.
  - Edit `RESULTS_CSV_PATH`, `REFERENCE_COLUMN` and `CANDIDATE_COLUMN` at the top of `metric.py` before running.
  - The script tokenizes, computes corpus BLEU-1..4, average ROUGE F1s, and CIDEr.

Notes, tips & troubleshooting
----------------------------
- GPU & memory: fine-tuning and some generation steps assume a GPU and sufficient VRAM. Use Colab Pro or a cloud instance if you don't have local GPUs.
- Tokenizer chat template: the LLaMA 3 chat template is used (see `tokenizer.apply_chat_template` calls). Keep tokenizer/model versions consistent across training and inference.
- Quantization: notebooks demonstrate 4-bit quantization via `bitsandbytes`. Be careful with compute dtype and tokenizers—mismatches cause generation errors.
- NLTK: `metric.py` expects the punkt tokenizer; it attempts to download automatically if missing.
- File paths: notebooks use `/content` paths for Colab. Update paths when running locally.

Next steps I can help with
-------------------------
- Create a `requirements.txt` that pins the main packages used in these notebooks.
- Convert key notebook cells to runnable scripts (`generate_captions.py`, `fine_tune.py`, `inference.py`).
- Add a small sample CSV and a short unit test to validate the end-to-end pipeline locally.

If you'd like, I can now generate a pinned `requirements.txt` for Task 2 and convert the captioning inference cell into a lightweight script.

---
Generated on: 2025-11-02
