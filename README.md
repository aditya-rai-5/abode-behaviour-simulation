
# Abode Behaviour Simulation — Project Overview

This repository contains two related tasks that together build a behaviour-simulation pipeline for social media marketing content. The system combines image captioning, prompt engineering, LLM fine-tuning, and a two-stage prediction pipeline to simulate and predict tweet performance.

Quick summary
-------------
- Task 1 (prediction): two-stage pipeline to predict tweet popularity.
	- Stage 1: Transformer-based classifier that buckets tweets (low / medium / high / viral).
	- Stage 2: LightGBM regressor that predicts real-valued likes (trained on log scale) using handcrafted features + Stage 1 probabilities.
	- Notebooks and files: `task 1/lightgbm+inference.ipynb`, `task 1/results_task1.ipynb`, `task 1/per_class_analysis.ipynb`, `task 1/README.md`.

- Task 2 (generation & captioning): image caption generation, prompt building, and fine-tuning an adapter for content generation.
	- Image captioning uses a Qwen2-VL model to produce detailed captions used as input to the content model.
	- Fine-tuning uses LoRA on a LLaMA base with 4-bit quantization support to train a content-generation adapter.
	- Notebooks and files: `task 2/image_captioning.ipynb`, `task 2/fine_tune_llama.ipynb`, `task 2/test pipeline.ipynb`, `task 2/data_cleaning.py`, `task 2/prompt.py`, `task 2/generate_captions.py`, `task 2/metric.py`, `task 2/README.md`.

Quick start (minimal)
---------------------
1. Create a virtual environment and install dependencies (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Task 1 (prediction): open `task 1/lightgbm+inference.ipynb` in Colab or Jupyter. Follow cells to mount drive (if Colab), generate features, train the LightGBM regressor, save schema and model, and run the two-stage inference helper.

3. Task 2 (captioning & generation): run `task 2/data_cleaning.py` to prepare data, generate captions with `task 2/generate_captions.py` or `task 2/image_captioning.ipynb`, then use `task 2/fine_tune_llama.ipynb` and `task 2/test pipeline.ipynb` to fine-tune and run content generation.

Evaluation
----------
- Task 1 evaluation: RMSE on real likes (expm1 of log predictions) and per-class classification metrics (precision/recall/F1).
- Task 2 evaluation: BLEU, ROUGE, and CIDEr implemented in `task 2/metric.py` (update the path to your results CSV before running).
