# Adobe Behaviour Simulation Challenge

This repository contains the complete solution for the Inter IIT Tech Meet Adobe Challenge, which is divided into two primary tasks:

1. **Task 1: Behavior Simulation:** To predict the number of 'likes' a tweet will receive (a regression task evaluated on RMSE).

2. **Task 2: Content Simulation:** To generate a realistic, context-aware marketing tweet (a generation task evaluated on BLEU, ROUGE, and CIDEr).

Our final solution consists of two separate, specialized pipelines that combine state-of-the-art transformer architectures to solve these distinct challenges.

## Task 1: Behavior Simulation (Predicting Likes)

Our iterative process led to a novel **2-Stage "Hybrid-Logic" Model** that effectively solves the core challenges of the dataset.

### Core Challenges

1. **Skewed Distribution:** Most tweets get few likes (0-100), while a tiny fraction go "viral" (10,000+).

2. **The "Exploding Error" Problem:** The RMSE metric is brutally punished by large errors. A small misprediction on a 10-like tweet (e.g., predicting 400) would devastate the model's score.

3. **Complex Data:** Success requires a deep, nuanced understanding of text, brand voice, and timing.

### Our Iterative Approach

1. **Attempt 1: `LightGBM + TF-IDF`:** This baseline model failed, producing a very high `Real RMSE` (\~3600). The "dumb" `TF-IDF` features couldn't capture the nuance of the text, and the model was highly susceptible to the "Exploding Error."

2. **Attempt 2: `LightGBM + BERT Embeddings`:** We replaced the "dumb" features with "smart" BERT embeddings. While this was a "smarter" model, it still produced a high `Real RMSE` (\~2800-3000). We learned that a single model forced to do two jobs (understand text *and* predict a precise number) was inefficient and still suffered from the "Exploding Error" problem.

3. **Attempt 3: The Classifier Breakthrough:** We changed the problem from regression to classification. We trained a `RoBERTa-base` Transformer to predict one of four buckets: **"Low" (0-100), "Medium" (101-1k), "High" (1k-10k), or "Viral" (10k+)**. This was a massive success, achieving \~80% accuracy and proving a Transformer could "sense" engagement levels.

### The Final, Winning Pipeline: A 2-Stage Hybrid Model

This pipeline divides the labor between two specialized models for maximum accuracy and stability.

#### **Stage 1: The "Expert Critic" (`RoBERTa` Classifier)**

* **Model:** `my_best_ROBERTA_model3` (a fine-tuned `RoBERTa-base` Transformer).

* **Job:** To act as the "smart" text-understanding engine. It reads the complex tweet and metadata.

* **Output:** An "expert opinion" in the form of **four probabilities**: `[prob_low, prob_medium, prob_high, prob_viral]`.

#### **Stage 2: The "Master Accountant" (`LightGBM` Regressor)**

* **Model:** A `LightGBM` Regressor.

* **Job:** To take the "expert opinion" from Stage 1 and predict the final *number*.

* **Features:** It was trained on a "master spreadsheet" combining:

  1. Simple manual features (`hour`, `dayofweek`, `sentiment`).

  2. The **four powerful probability features** from the Stage 1 Classifier.

* **Result:** This 2-stage model was our most accurate by far, achieving a **Log RMSE of 0.7347** (a 16.5% improvement over our previous best).

#### **The Final Fix: "Hybrid-Logic" Inference**

To get the best possible `Real RMSE`, we use a final rule during prediction:

1. **Run Stage 1 (Classifier)** to get the 4 probabilities.

2. **Ask One Question:** Is `prob_low` > 90%?

   * **IF YES (Boring Tweet):** We **STOP** and manually predict a safe, low number (e.g., **50 likes**). This makes the "Exploding Error" *impossible*.

   * **IF NO (Potential):** We **proceed to Stage 2** and use the "Master Accountant" to get its highly accurate `log(likes)` prediction (e.g., 9.5), which we convert back to a real number (e.g., 13,350 likes).

This pipeline is fast, novel, and highly accurate, using the Transformer's "brain" to fix the "Exploding Error" problem.

## Task 2: Content Simulation (Generating Tweets)

This task uses a sophisticated multi-stage pipeline to synthesize context-aware social media content by interpreting visual data (images) and leveraging a fine-tuned Large Language Model (LLM).

### Model Architecture

| Component | Base Model | Role |
| :--- | :--- | :--- |
| **Vision-Language Model (VLM)** | `Qwen2-VL-2B-Instruct` | Generates detailed, objective captions from image URLs. |
| **Large Language Model (LLM)** | `LLAMA 3.2-3B-Instruct` | Generates the final, context-aware marketing tweet. |

### Generation Pipeline

1. **Data Cleaning:** Input data (dates, likes, URLs) is preprocessed and standardized.

2. **Image Captioning:** The `Qwen2-VL` model is used to analyze the image URL and generate a rich, descriptive caption (e.g., "A high-quality photo of a new product on a wooden table").

3. **Prompt Engineering:** We create a custom prompt that combines all available context for the LLM:

   * The generated image caption from Stage 2.

   * Brand identity (username, company).

   * Temporal information (date, time).

   * User engagement metrics (likes).

4. **LLM Fine-Tuning:** The `LLAMA 3.2-3B` model is fine-tuned using LoRA on these custom-engineered prompts to teach it how to generate marketing tweets.

5. **Inference:** The `test_pipeline.ipynb` notebook executes this full process to generate the final content.

### Evaluation

The quality of the generated tweets is assessed using standard NLG metrics:

* **BLEU (1-4):** Measures n-gram similarity to reference tweets.

* **ROUGE (1, 2, L):** Measures n-gram and subsequence overlap.

* **CIDEr:** Used to evaluate the quality of the intermediate image captions.

## Future Work: SOTA 3-Stage "Conditional" Pipeline

Our analysis showed that both tasks could be improved by intelligently combining their components. Our proposed SOTA approach is a 3-stage "conditional" pipeline to solve Task 1 with maximum efficiency.

1. **Stage 1: Fast Triage (RoBERTa Classifier):** Our existing `my_best_ROBERTA_model3` runs on *all* tweets to provide the 4 engagement probabilities.

2. **Stage 2: Conditional Image Analysis (VLM):** A vision model (like `Gemini` or `Qwen2-VL`) is **only activated** if the Stage 1 prediction for "High" or "Viral" is above a certain confidence threshold. This saves massive computational cost.

3. **Stage 3: Final Regressor (LightGBM):** Our existing `LightGBM` "Accountant" is re-trained on an even richer dataset that now includes:

   * Manual Features (`hour`, `sentiment`...)

   * Stage 1 Probabilities (`prob_low`...)

   * **New Vision Feature:** The image caption or embedding (or a "0" if Stage 2 was skipped).

This hybrid approach is highly efficient by skipping expensive image processing on 80% of the "Low" tweets, while still using the powerful vision signal on the "High" and "Viral" tweets where it matters most.

## Repository Structure

* `data_cleaning.py`: Scripts for preprocessing and standardizing input data.

* `image_captioning.ipynb`: Notebook for Stage 2 of Task 2, using `Qwen2-VL` to generate captions.

* `prompt.py`: Utility for creating the custom prompt templates.

* `fine_tune_llama.ipynb`: Notebook for fine-tuning the `LLAMA` model for Task 2.

* `test_pipeline.ipynb`: Final inference notebook for generating tweet content (Task 2).

* `metric.py`: Scripts for calculating BLEU, ROUGE, and CIDEr scores.

* *(Other notebooks for Task 1, e.g., `Task_1_Training.ipynb`, `Task_1_Inference.ipynb`)*
