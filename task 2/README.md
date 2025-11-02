# Abode Behaviour Simulation

This repository implements an advanced social media marketing content generation system using Large Language Models (LLMs) and computer vision. It simulates marketing behavior by generating contextually appropriate content from images, metadata, and brand identity.

## Project Structure

### Task 2 — Image Captioning & Content Generation (Fine-tune / Inference)

This folder contains notebooks and scripts used for Task 2: generating descriptive image captions, building prompts, and fine-tuning a LLaMA-based content-generation adapter.

### Contents

- `data_cleaning.py` — lightweight preprocessing: date parsing, media URL extraction, media type tagging, column renaming, and small sanity checks. Update input/output filenames before running.
- `image_captioning.ipynb` — loads Qwen2-VL model (quantized) and generates detailed captions for images (URLs or local files). Supports batching, intermediate saves, and thumbnailing.
- `fine_tune_llama.ipynb` — fine-tuning notebook that:
  - Prepares prompts using `prompt.py` logic
  - Applies LLaMA chat templates
  - Configures LoRA and 4-bit quantization
  - Creates dataset for SFTTrainer and trains/saves an adapter
- `prompt.py` — prompt-building utilities for training and inference
- `metric.py` — computes BLEU, ROUGE, CIDEr on CSV with `content` (reference) and `generated_content` (candidate). Update `RESULTS_CSV_PATH` before running.
- `test pipeline.ipynb` — end-to-end inference:
  - Loads fine-tuned adapter + base model
  - Uses image_captioning outputs as inputs
  - Creates final input prompts and runs batched generation
  - Saves `generated_content` to CSV

## Data Expectations

Input CSV should contain:

- `date` — timestamp (parseable by pandas)
- `likes` — historical likes count (training only)
- `username` — handle (brand identity)
- `inferred company` or `inferred_company` — inferred brand name
- `media` / `media_url` — image URL or descriptor
- `content` — original tweet text (for training/evaluation)

## Quick Start (Colab)

1. Open `image_captioning.ipynb` or `fine_tune_llama.ipynb`
2. Mount Google Drive
3. Install required libraries (`bitsandbytes`, `transformers`, `trl`, etc.)
4. Update paths (e.g., `/content/test_dataset.csv`) and run cells sequentially

## Quick Start (Local / Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install pandas numpy torch transformers bitsandbytes accelerate peft trl datasets pillow rouge-score pycocoevalcap nltk
```

Run preprocessing:

```powershell
python "task 2\data_cleaning.py"
```

Run image captioning to generate `generated_caption`.
Fine-tune using `fine_tune_llama.ipynb`. Run `test pipeline.ipynb` to generate final content.

## Evaluation

Run `metric.py` after updating:

```python
RESULTS_CSV_PATH
REFERENCE_COLUMN
CANDIDATE_COLUMN
```

Metrics:
- BLEU-1..4
- ROUGE-1, ROUGE-2, ROUGE-L
- CIDEr

## Notes & Troubleshooting

- GPU recommended for fine-tuning and generation
- Keep tokenizer and model versions consistent
- Quantization dtype mismatches may cause errors
- NLTK punkt tokenizer downloaded automatically if missing
- Update paths when running locally

## Model Architecture

- Base Model: LLaMA 3.2-3B-Instruct
- Image Model: Qwen2-VL-2B-Instruct
- LoRA fine-tuning with 4-bit quantization
- Custom marketing prompt templates

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- TRL
- bitsandbytes
- Pandas
- Pillow
- NLTK

## License

See the [LICENSE](LICENSE) file for details.