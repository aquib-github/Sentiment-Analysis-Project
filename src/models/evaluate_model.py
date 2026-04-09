"""
Model Evaluation Module.

Computes classification metrics (accuracy, precision, recall, F1)
and confusion matrices for trained models.
"""

from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    label_names: List[str],
) -> Dict[str, Any]:
    """
    Evaluate a trained pipeline on the test set.

    Args:
        pipeline:    Fitted sklearn Pipeline.
        X_test:      Test text data.
        y_test:      True labels.
        label_names: Human-readable class names.

    Returns:
        Dict with keys: accuracy, precision, recall, f1, confusion_matrix,
        classification_report, predictions.
    """
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, target_names=label_names, output_dict=True,
        ),
        "predictions": y_pred,
    }

    logger.info(
        "Evaluation — Accuracy: %.4f | F1: %.4f",
        metrics["accuracy"], metrics["f1"],
    )
    return metrics


def evaluate_all_models(
    results: Dict[str, Any],
    X_test: pd.Series,
    y_test: pd.Series,
    label_names: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate every trained model and return metrics for each.

    Args:
        results:     Output from ``train_all_models``.
        X_test:      Test text data.
        y_test:      True labels.
        label_names: Human-readable class names.

    Returns:
        Dict mapping model name → evaluation metrics dict.
    """
    evaluations: Dict[str, Dict[str, Any]] = {}

    for name, res in results.items():
        logger.info("Evaluating %s …", name)
        evaluations[name] = evaluate_model(
            pipeline=res["pipeline"],
            X_test=X_test,
            y_test=y_test,
            label_names=label_names,
        )

    return evaluations


def build_comparison_dataframe(
    evaluations: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Build a summary DataFrame comparing all models.

    Args:
        evaluations: Output from ``evaluate_all_models``.

    Returns:
        DataFrame with columns: Model, Accuracy, Precision, Recall, F1 Score.
    """
    rows = []
    for name, metrics in evaluations.items():
        rows.append({
            "Model": name,
            "Accuracy": round(metrics["accuracy"] * 100, 2),
            "Precision": round(metrics["precision"] * 100, 2),
            "Recall": round(metrics["recall"] * 100, 2),
            "F1 Score": round(metrics["f1"] * 100, 2),
        })

    return pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
