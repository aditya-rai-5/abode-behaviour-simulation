<<<<<<< HEAD
=======
   # Abode Content Simulation

This project implements a multi-stage machine learning pipeline to generate simulated marketing tweets. The core idea is to use a Vision-Language Model (VLM) to generate rich, objective captions for images, and then feed these captions (along with other metadata like username, likes, etc.) into a fine-tuned Large Language Model (LLM) to generate realistic tweet content.

The pipeline is divided into six main stages: data cleaning, image captioning, training prompt preparation, LLM fine-tuning, inference, and evaluation.
## Project Structure

```
├── task 2/
│   ├── data_cleaning.py          # Data preprocessing and cleaning
│   ├── fine_tune_llama.ipynb     # LLaMA model fine-tuning
│   ├── image_captioning.ipynb    # Image caption generation using Qwen-VL
│   ├── metric.py                 # Evaluation metrics (BLEU, ROUGE, CIDEr)
│   ├── prompt.py                 # Prompt engineering and generation
│   └── test pipeline.ipynb       # Testing and inference pipeline
└── test_dataset/                 # Test data directory
```

## Features

1. **Data Preprocessing**
   - Cleans and standardizes input data
   - Handles multiple data fields: dates, likes, usernames, media types
   - Extracts media URLs and categorizes content types

2. **Image Captioning**
   - Uses Qwen2-VL-2B-Instruct model for image understanding
   - Generates detailed captions for marketing images
   - Supports batch processing of images

3. **Content Generation**
   - Fine-tuned LLaMA model for marketing content
   - Context-aware generation considering:
     - Brand identity
     - Media context
     - Temporal information
     - User engagement metrics

4. **Model Training**
   - Implementation of LoRA fine-tuning
   - Quantization support for efficient training
   - Custom prompt engineering for marketing context

5. **Evaluation Metrics**
   - BLEU score for content similarity
   - ROUGE metrics for text quality
   - CIDEr score for caption evaluation

## Setup and Usage

1. **Environment Setup**
   ```bash
   pip install -U bitsandbytes
   pip install transformers accelerate
   pip install trl
   ```

2. **Data Preparation**
   - Run `data_cleaning.py` to process your input data
   - Ensure proper formatting of input CSV files

3. **Image Captioning**
   - Use `image_captioning.ipynb` for generating image descriptions
   - Supports both URL and local image inputs

4. **Model Fine-tuning**
   - Follow `fine_tune_llama.ipynb` for model training
   - Configure hyperparameters as needed

5. **Testing**
   - Use `test pipeline.ipynb` for inference
   - Evaluate results using `metric.py`

## Model Architecture

- Base Model: LLaMA 3.2-3B-Instruct
- Image Model: Qwen2-VL-2B-Instruct
- Training: LoRA fine-tuning with 4-bit quantization
- Custom prompt templates for marketing context

## Evaluation

The system is evaluated using multiple metrics:
- BLEU-1 through BLEU-4 scores
- ROUGE-1, ROUGE-2, and ROUGE-L metrics
- CIDEr scoring for caption quality

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- NLTK
- Pandas
- PIL (Pillow)
- bitsandbytes
- TRL (Transformer Reinforcement Learning)

## License

See the [LICENSE](LICENSE) file for details.
>>>>>>> aa555f26f0753f1622494881475579095f039624
