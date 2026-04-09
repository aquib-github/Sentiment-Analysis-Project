"""
Feature Engineering Module.

Handles TF-IDF vectorization and label encoding, wrapping them
inside an sklearn Pipeline for reproducibility.
"""

from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def encode_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Encode the target sentiment column into numeric labels.

    Args:
        df: DataFrame containing the ``sentiment`` column.

    Returns:
        Tuple of (updated DataFrame with ``label`` column, fitted LabelEncoder).
    """
    le = LabelEncoder()
    df = df.copy()
    df["label"] = le.fit_transform(df[config.TARGET_COLUMN])

    logger.info("Label encoding complete — classes: %s", list(le.classes_))
    return df, le


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """
    Create a TF-IDF vectorizer using project-wide configuration.

    Returns:
        An unfitted ``TfidfVectorizer`` instance.
    """
    return TfidfVectorizer(**config.TFIDF_PARAMS)


def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Perform a stratified train / validation / test split.

    Split ratio: 80 % train · 10 % validation · 10 % test.

    Args:
        df: DataFrame containing ``clean_text`` and ``label`` columns.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X = df[config.CLEAN_TEXT_COLUMN]
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=config.VALIDATION_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_temp,
    )

    logger.info(
        "Data split — Train: %d | Val: %d | Test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
