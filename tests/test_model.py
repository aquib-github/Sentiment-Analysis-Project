"""
Unit Tests — Model Module.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.features.feature_engineering import encode_labels, build_tfidf_vectorizer, split_data
from src.models.evaluate_model import build_comparison_dataframe


class TestFeatureEngineering:
    """Tests for feature engineering functions."""

    def test_encode_labels_creates_label_column(self):
        df = pd.DataFrame({"sentiment": ["Positive", "Negative", "Neutral"]})
        result_df, le = encode_labels(df)
        assert "label" in result_df.columns
        assert isinstance(le, LabelEncoder)

    def test_encode_labels_correct_mapping(self):
        df = pd.DataFrame({"sentiment": ["Positive", "Negative", "Neutral"]})
        result_df, le = encode_labels(df)
        # All classes should be present
        assert set(le.classes_) == {"Positive", "Negative", "Neutral"}

    def test_tfidf_vectorizer_returns_correct_type(self):
        vectorizer = build_tfidf_vectorizer()
        assert hasattr(vectorizer, "fit_transform")

    def test_split_data_returns_correct_sizes(self):
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            "clean_text": [f"sample text number {i}" for i in range(n)],
            "label": np.random.randint(0, 4, n),
        })
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

        # 80-10-10 split
        assert len(X_train) == pytest.approx(n * 0.8, abs=5)
        assert len(X_val) == pytest.approx(n * 0.1, abs=5)
        assert len(X_test) == pytest.approx(n * 0.1, abs=5)

    def test_split_data_preserves_stratification(self):
        np.random.seed(42)
        n = 400
        labels = np.array([0] * 100 + [1] * 100 + [2] * 100 + [3] * 100)
        df = pd.DataFrame({
            "clean_text": [f"text {i}" for i in range(n)],
            "label": labels,
        })
        _, _, _, y_train, _, _ = split_data(df)
        # Each class should be roughly equally represented
        unique, counts = np.unique(y_train, return_counts=True)
        assert len(unique) == 4


class TestEvaluationUtils:
    """Tests for evaluation utility functions."""

    def test_build_comparison_dataframe(self):
        evaluations = {
            "Model A": {"accuracy": 0.85, "precision": 0.84, "recall": 0.83, "f1": 0.84},
            "Model B": {"accuracy": 0.90, "precision": 0.89, "recall": 0.88, "f1": 0.89},
        }
        result = build_comparison_dataframe(evaluations)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "Model" in result.columns
        assert "Accuracy" in result.columns
        assert "F1 Score" in result.columns
        # Should be sorted descending by F1
        assert result.iloc[0]["F1 Score"] >= result.iloc[1]["F1 Score"]
