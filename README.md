# ABSA Hotel Reviews: English vs Spanish Models

Darragh Kerins and Ander Peña

## Overview

This project compares different approaches to Aspect-Based Sentiment Analysis (ABSA) for hotel reviews in Spanish. We train and evaluate four mT5 models:

- **Model A**: English baseline (M-ABSA hotel_en)
- **Model B**: Spanish with machine-translated data (M-ABSA hotel_es)
- **Model C**: Spanish with synthetic LLM-generated data
- **Model D**: Spanish with combined translated + synthetic data

The main goal is to test whether synthetic Spanish data outperforms machine-translated Spanish for training ABSA models.

## Repository Structure

```
Project/
├── notebooks/
│   └── absa_pipeline.ipynb    # Full training and evaluation pipeline
├── Data_synthetic/            # Synthetic Spanish hotel reviews
│   ├── synthetic_train_1.txt
│   ├── synthetic_train_2.txt
│   └── ...
├── absa/                      # Core Python modules (if needed)
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   └── train.py
└── requirements.txt             # Dependencies

```

## Data Sources

- **English**: M-ABSA dataset (hotel_en config) - 1,255 train, 584 test
- **Spanish translated**: M-ABSA dataset (hotel_es config) - machine-translated from English
- **Spanish synthetic**: LLM-generated using Claude - stored in Data_synthetic/

## Running the Pipeline

1. Open `notebooks/absa_pipeline.ipynb` in Google Colab or Jupyter
2. Run cells sequentially from top to bottom
3. Models are saved to `checkpoints/` directory
4. Evaluation results print at the end with F1 comparison table

## Requirements

```
torch
transformers
datasets
accelerate
numpy
```

Install with: `pip install -r requirements.txt`

## Key Findings

The notebook compares F1 scores across all models and answers:
- Does synthetic Spanish data beat machine-translated Spanish?
- How well does English zero-shot transfer to Spanish?
- Does combining data sources improve results?

Check the results table at the end of the notebook for the comparison.
