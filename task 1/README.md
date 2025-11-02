
# Task 1 — Behaviour Simulation & Tweet Popularity Prediction

This folder contains the code and notebooks for Task 1 of the Abode Behaviour Simulation project: building a two-stage pipeline that predicts tweet popularity (bucket and real-valued likes) from tweet text, metadata and image-derived signals.

Summary
-------
- Stage 1: a transformer-based classifier (your winning RoBERTa/DistilBERT model) that predicts a discrete bucket for tweet popularity (e.g., low / medium / high / viral).
- Stage 2: a LightGBM regressor that predicts the real number of likes (trained on the log scale). The regressor uses handcrafted features (text length, time, sentiment, categorical encodings) plus the Stage 1 class probabilities.

Files and notebooks
-------------------
- `results_task1.ipynb` — analysis and evaluation cells used to run Stage 1 -> Stage 2 pipeline and produce the "highlight reel" of best predictions.
- `lightgbm+inference.ipynb` — a complete, corrected notebook that shows how to
	- generate features
	- train the LightGBM Stage 2 regressor
	- save the regressor and a schema (dtypes)
	- run a two-stage inference pipeline (classifier -> regressor)
	- includes fixes for categorical handling and safe NaN filling
- `per_class_analysis.ipynb` — runs a per-class report for your Stage 1 classifier (accuracy / F1 / classification_report) on the recreated validation set.

Data expectations
-----------------
Input data is expected as a tabular CSV / Excel with at least these columns (names used by the notebooks):

- `content` — the tweet text
- `likes` — numeric likes (used to create buckets and to train regressor)
- `date` (or `dates`) — timestamp for the tweet
- `media` — optional media information / URL
- `company` (or `inferred company`) — brand / company name
- `username` — account name

The notebooks recreate an 80/20 split from the full training data to match experiment conditions used in training/validation. Keep original categories (company/username) accessible for factorization.

How the two-stage pipeline works (contract)
-----------------------------------------
- Inputs: single tweet row with text + metadata (company, username, date/time, media flag).
- Output: predicted likes (real-valued, non-negative) and optional class bucket.
- Error modes: missing required columns will raise errors. Unknown companies/usernames are handled via safe defaults (-1 or NaNs left for LightGBM categorical handling).

Quick setup (Colab / local)
---------------------------
Colab (recommended for GPU/TPU): open the notebooks and run cells. Notebooks already include mount & pip install commands.

Local (Windows / CPU or GPU):

1. Create a virtual environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install pandas numpy scikit-learn lightgbm transformers datasets joblib vaderSentiment openpyxl
```

2. Place your data file in Google Drive (if using the Colab notebooks) or update paths in notebooks to point to local files.

Running training & inference
----------------------------
- Feature generation and transformer probability extraction: run the feature-generation cell in `lightgbm+inference.ipynb` (this calls your Stage 1 classifier to create probability columns). This is the slowest step.
- Train the LightGBM regressor: run the training cell (uses `log1p(likes)` as target). Save the model using `joblib.dump()` as shown.
- Save the schema: the notebook saves a `my_data_schema.pkl` (dtypes) which is used by the inference pipeline to recreate categorical dtypes exactly.
- Final inference: use the provided `predict_likes_twostage_hybrid` helper in `lightgbm+inference.ipynb` to produce predictions for new tweets.

Evaluation
----------
- The final Task 1 score is RMSE on the real likes scale (computed after exponentiating from log predictions). The notebooks compute RMSE using sklearn's `mean_squared_error`.
- Per-class classification reports (precision, recall, F1) for the Stage 1 classifier are in `per_class_analysis.ipynb`.

Tips & common pitfalls
----------------------
- Keep the training schema (the saved dtype object) together with the saved regressor. LightGBM needs consistent categorical types for correct predictions.
- Do not fill categorical columns with numeric defaults (e.g., 0) before prediction — the notebook demonstrates safe handling (leave NaNs for categorical fields; fill numeric columns only).
- The transformer inference step requires a GPU for speed; plan accordingly or run in small batches locally.

Next steps / improvements
-------------------------
- Automate the full pipeline into a single script (`train.py` / `predict.py`) so it can run outside notebooks.
- Add unit tests for the feature generation functions and a small sample dataset to validate end-to-end behavior.
- Add a lightweight CLI for batch inference using saved models and schema.

If you want, I can:
- add a `requirements.txt` with pinned versions,
- convert the essential inference cells into a runnable `predict.py` script,
- or create the single-file README for the repository root describing both Task 1 and Task 2 (image captioning + content generation).

---
