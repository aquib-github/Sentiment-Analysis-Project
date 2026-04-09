"""
Configuration Module for Sentiment Analysis Project.

Centralizes all configuration parameters including file paths,
model hyperparameters, and application settings.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# BASE DIRECTORIES
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
ASSETS_DIR = BASE_DIR / "assets" / "images"

# ──────────────────────────────────────────────────────────────
# FILE PATHS
# ──────────────────────────────────────────────────────────────
RAW_DATASET_PATH = RAW_DATA_DIR / "twitter_training.csv"
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "processed_data.csv"

TRAINED_MODEL_PATH = MODEL_DIR / "trained_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
METRICS_PATH = MODEL_DIR / "metrics.pkl"

LOG_FILE_PATH = LOG_DIR / "app.log"

# ──────────────────────────────────────────────────────────────
# DATASET CONFIGURATION
# ──────────────────────────────────────────────────────────────
DATASET_COLUMNS = ["id", "topic", "sentiment", "text"]
TARGET_COLUMN = "sentiment"
TEXT_COLUMN = "text"
CLEAN_TEXT_COLUMN = "clean_text"

# ──────────────────────────────────────────────────────────────
# MODEL CONFIGURATION
# ──────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.5  # 50% of the 20% test split → 10% val, 10% test

# TF-IDF parameters
TFIDF_PARAMS = {
    "max_features": 20000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
}

# Model hyperparameters for GridSearchCV
MODEL_PARAMS = {
    "Naive Bayes": {
        "classifier__alpha": [0.01, 0.1, 0.5, 1.0],
    },
    "Logistic Regression": {
        "classifier__C": [0.5, 1.0, 5.0, 10.0],
        "classifier__max_iter": [1000],
    },
    "SVM": {
        "classifier__C": [0.5, 1.0, 2.0],
        "classifier__max_iter": [2000],
    },
}

# Cross-validation folds
CV_FOLDS = 5

# ──────────────────────────────────────────────────────────────
# SENTIMENT EMOJI MAPPING
# ──────────────────────────────────────────────────────────────
SENTIMENT_EMOJIS = {
    "Positive": "😊",
    "Negative": "😠",
    "Neutral": "😐",
    "Irrelevant": "🤷",
}

SENTIMENT_COLORS = {
    "Positive": "#2ecc71",
    "Negative": "#e74c3c",
    "Neutral": "#3498db",
    "Irrelevant": "#f39c12",
}

# ──────────────────────────────────────────────────────────────
# WORDCLOUD COLORMAPS
# ──────────────────────────────────────────────────────────────
WORDCLOUD_CMAPS = {
    "Positive": "Greens",
    "Negative": "Reds",
    "Neutral": "Blues",
    "Irrelevant": "Oranges",
}

# ──────────────────────────────────────────────────────────────
# ENSURE DIRECTORIES EXIST
# ──────────────────────────────────────────────────────────────
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR, ASSETS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
