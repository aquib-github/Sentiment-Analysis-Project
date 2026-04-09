"""
Prediction Module.

Provides functions to predict sentiment from raw text using
previously trained and saved model artifacts.
"""

from typing import Dict, Any
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

import config
from src.data.data_preprocessing import clean_text
from src.utils.helpers import load_artifact
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_trained_pipeline() -> Pipeline:
    """
    Load the trained model pipeline from disk.

    Returns:
        Fitted sklearn Pipeline.

    Raises:
        FileNotFoundError: If the model artifact is missing.
    """
    return load_artifact(config.TRAINED_MODEL_PATH)


def load_label_encoder() -> LabelEncoder:
    """
    Load the label encoder from disk.

    Returns:
        Fitted LabelEncoder.
    """
    return load_artifact(config.LABEL_ENCODER_PATH)


def predict_sentiment(
    text: str,
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
) -> Dict[str, Any]:
    """
    Predict the sentiment of a single text input.

    Args:
        text:          Raw user input text.
        pipeline:      Fitted Pipeline (TF-IDF + classifier).
        label_encoder: Fitted LabelEncoder for inverse transform.

    Returns:
        Dict with keys:
            - ``original_text``: the raw input
            - ``cleaned_text``: after preprocessing
            - ``predicted_label``: human-readable sentiment
            - ``predicted_index``: numeric label
            - ``confidence``: probability or decision-function score
            - ``emoji``: sentiment emoji
    """
    cleaned = clean_text(text)
    pred_idx = pipeline.predict([cleaned])[0]
    label = label_encoder.inverse_transform([pred_idx])[0]

    # Attempt to get probability/confidence
    confidence = _get_confidence(pipeline, cleaned, pred_idx)

    emoji = config.SENTIMENT_EMOJIS.get(label, "❓")

    logger.info("Prediction: '%s' → %s %s (conf: %.2f)", text[:50], label, emoji, confidence)

    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "predicted_label": label,
        "predicted_index": int(pred_idx),
        "confidence": confidence,
        "emoji": emoji,
    }


def _get_confidence(pipeline: Pipeline, cleaned_text: str, pred_idx: int) -> float:
    """
    Extract a confidence score from the pipeline's classifier.

    Uses ``predict_proba`` if available, otherwise falls back to
    ``decision_function``. Returns 0.0 if neither is available.
    """
    classifier = pipeline.named_steps.get("classifier")

    try:
        if hasattr(classifier, "predict_proba"):
            tfidf = pipeline.named_steps["tfidf"]
            vec = tfidf.transform([cleaned_text])
            proba = classifier.predict_proba(vec)[0]
            return float(np.max(proba))
        elif hasattr(classifier, "decision_function"):
            tfidf = pipeline.named_steps["tfidf"]
            vec = tfidf.transform([cleaned_text])
            scores = classifier.decision_function(vec)[0]
            # Normalize via softmax approximation
            exp_scores = np.exp(scores - np.max(scores))
            softmax = exp_scores / exp_scores.sum()
            return float(softmax[pred_idx])
    except Exception:
        pass

    return 0.0
