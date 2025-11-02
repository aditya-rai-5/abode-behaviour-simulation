import pandas as pd
import nltk
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
import numpy as np

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt')

RESULTS_CSV_PATH = ""
REFERENCE_COLUMN = "content" 
CANDIDATE_COLUMN = "generated_content"

print(f"Loading results file from: {RESULTS_CSV_PATH}")
try:
    df = pd.read_csv(RESULTS_CSV_PATH)
except FileNotFoundError:
    print(f"ERROR: File not found at {RESULTS_CSV_PATH}")
    exit()
except Exception as e:
    print(f"ERROR: Could not read file. {e}")
    exit()

df = df.dropna(subset=[REFERENCE_COLUMN, CANDIDATE_COLUMN])
df = df.reset_index(drop=True)

if df.empty:
    print("ERROR: No valid data found. Check column names.")
    print(f"Expected: '{REFERENCE_COLUMN}' and '{CANDIDATE_COLUMN}'")
    exit()

print(f"Loaded {len(df)} rows for evaluation.")


def tokenize(text):
    """Simple whitespace tokenizer aur lowercase."""
    return str(text).lower().split()

references_tokenized = df[REFERENCE_COLUMN].apply(lambda x: [tokenize(x)]).tolist()
candidates_tokenized = df[CANDIDATE_COLUMN].apply(tokenize).tolist()

references_raw = df[REFERENCE_COLUMN].astype(str).tolist()
candidates_raw = df[CANDIDATE_COLUMN].astype(str).tolist()

print("\n--- 1. Calculating BLEU Scores ---")
bleu1 = corpus_bleu(references_tokenized, candidates_tokenized, weights=(1, 0, 0, 0))
bleu2 = corpus_bleu(references_tokenized, candidates_tokenized, weights=(0.5, 0.5, 0, 0))
bleu3 = corpus_bleu(references_tokenized, candidates_tokenized, weights=(0.333, 0.333, 0.333, 0))
bleu4 = corpus_bleu(references_tokenized, candidates_tokenized, weights=(0.25, 0.25, 0.25, 0.25))

print(f"BLEU-1 (Cumulative): {bleu1:.4f}")
print(f"BLEU-2 (Cumulative): {bleu2:.4f}")
print(f"BLEU-3 (Cumulative): {bleu3:.4f}")
print(f"BLEU-4 (Cumulative): {bleu4:.4f}")

print("\n--- 2. Calculating ROUGE Scores ---")
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge1_f, rouge2_f, rougeL_f = [], [], []

for ref, cand in zip(references_raw, candidates_raw):
    scores = scorer.score(ref, cand)
    rouge1_f.append(scores['rouge1'].fmeasure)
    rouge2_f.append(scores['rouge2'].fmeasure)
    rougeL_f.append(scores['rougeL'].fmeasure)

print(f"ROUGE-1 (F1-Score Avg): {np.mean(rouge1_f):.4f}")
print(f"ROUGE-2 (F1-Score Avg): {np.mean(rouge2_f):.4f}")
print(f"ROUGE-L (F1-Score Avg): {np.mean(rougeL_f):.4f}")


print("\n--- 3. Calculating CIDEr Score (Updated Method) ---")
gts = {i: [ref] for i, ref in enumerate(references_raw)}
res = {i: [cand] for i, cand in enumerate(candidates_raw)}

cider_scorer = Cider() 

cider_avg_score, cider_scores = cider_scorer.compute_score(gts, res)

print(f"CIDEr (Average Score): {cider_avg_score:.4f}")

print("\n--- Evaluation Complete ---")