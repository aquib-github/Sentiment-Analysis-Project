"""
Model Training Module.

Builds sklearn Pipelines for each classifier, runs GridSearchCV
with cross-validation, selects the best model, and persists artifacts.
"""

from typing import Dict, Any, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import config
from src.utils.logger import get_logger
from src.utils.helpers import save_artifact

logger = get_logger(__name__)


def _build_pipelines() -> Dict[str, Pipeline]:
    """
    Construct an sklearn Pipeline for each classifier.

    Returns:
        Dict mapping model name → Pipeline.
    """
    tfidf_params = config.TFIDF_PARAMS

    pipelines: Dict[str, Pipeline] = {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("classifier", MultinomialNB()),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("classifier", LogisticRegression(
                random_state=config.RANDOM_STATE,
                max_iter=1000,
            )),
        ]),
        "SVM": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("classifier", LinearSVC(
                random_state=config.RANDOM_STATE,
                max_iter=2000,
            )),
        ]),
    }
    return pipelines


def train_all_models(
    X_train: pd.Series,
    y_train: pd.Series,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Train all models using GridSearchCV with stratified k-fold CV.

    Args:
        X_train:           Training text data.
        y_train:           Training labels.
        progress_callback: Optional callable(step, total, message) for UI updates.

    Returns:
        Dict mapping model name → {
            "pipeline": fitted Pipeline,
            "best_params": dict,
            "best_cv_score": float,
        }
    """
    pipelines = _build_pipelines()
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    results: Dict[str, Any] = {}

    total = len(pipelines)
    for idx, (name, pipeline) in enumerate(pipelines.items(), 1):
        logger.info("Training %s (%d/%d) …", name, idx, total)

        if progress_callback:
            progress_callback(idx, total, f"Training {name} …")

        param_grid = config.MODEL_PARAMS.get(name, {})

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)

        results[name] = {
            "pipeline": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "best_cv_score": grid_search.best_score_,
        }

        logger.info(
            "%s — best CV F1: %.4f | params: %s",
            name, grid_search.best_score_, grid_search.best_params_,
        )

    return results


def select_best_model(results: Dict[str, Any]) -> Tuple[str, Pipeline]:
    """
    Select the model with the highest cross-validation F1 score.

    Args:
        results: Output from ``train_all_models``.

    Returns:
        Tuple of (best model name, fitted Pipeline).
    """
    best_name = max(results, key=lambda n: results[n]["best_cv_score"])
    best_pipeline = results[best_name]["pipeline"]
    logger.info("Best model selected → %s (CV F1: %.4f)", best_name, results[best_name]["best_cv_score"])
    return best_name, best_pipeline


def save_trained_artifacts(pipeline: Pipeline, label_encoder) -> None:
    """
    Save the trained pipeline and label encoder to disk.

    Args:
        pipeline:      The best fitted Pipeline (includes TF-IDF + classifier).
        label_encoder: Fitted LabelEncoder.
    """
    save_artifact(pipeline, config.TRAINED_MODEL_PATH)
    save_artifact(label_encoder, config.LABEL_ENCODER_PATH)
    logger.info("All training artifacts saved to %s", config.MODEL_DIR)
