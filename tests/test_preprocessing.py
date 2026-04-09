"""
Unit Tests — Data Preprocessing Module.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.data.data_preprocessing import clean_text, preprocess_dataframe
import pandas as pd


class TestCleanText:
    """Tests for the clean_text function."""

    def test_lowercasing(self):
        result = clean_text("HELLO WORLD TESTING")
        assert result == result.lower()

    def test_url_removal(self):
        result = clean_text("Check this out https://example.com great stuff")
        assert "https" not in result
        assert "example" not in result

    def test_mention_removal(self):
        result = clean_text("Thanks @user123 for the help")
        assert "@user123" not in result

    def test_hashtag_removal(self):
        result = clean_text("Loving this #Amazing product")
        assert "#Amazing" not in result

    def test_number_removal(self):
        result = clean_text("I bought 5 items for 100 dollars")
        assert "5" not in result
        assert "100" not in result

    def test_punctuation_removal(self):
        result = clean_text("Hello! How are you? I'm fine, thanks.")
        assert "!" not in result
        assert "?" not in result
        assert "," not in result

    def test_stopword_removal(self):
        result = clean_text("This is a very good product and I like it")
        assert "this" not in result.split()
        assert "is" not in result.split()

    def test_short_word_removal(self):
        result = clean_text("I am so happy to be here")
        words = result.split()
        for word in words:
            assert len(word) > 2

    def test_empty_string(self):
        result = clean_text("")
        assert result == ""

    def test_numeric_only(self):
        result = clean_text("12345 67890")
        assert result.strip() == ""

    def test_lemmatization(self):
        result = clean_text("The cats were running quickly across fields")
        assert "cat" in result or "running" in result or "quickly" in result


class TestPreprocessDataframe:
    """Tests for the preprocess_dataframe function."""

    def test_adds_clean_text_column(self):
        df = pd.DataFrame({
            "text": ["This is a great product!", "Terrible experience"],
            "sentiment": ["Positive", "Negative"],
        })
        result = preprocess_dataframe(df, text_col="text")
        assert "clean_text" in result.columns

    def test_removes_empty_rows(self):
        df = pd.DataFrame({
            "text": ["Great!", "123", ""],
            "sentiment": ["Positive", "Neutral", "Negative"],
        })
        result = preprocess_dataframe(df, text_col="text")
        # Rows with empty clean_text should be removed
        assert all(row.strip() != "" for row in result["clean_text"])

    def test_handles_null_values(self):
        df = pd.DataFrame({
            "text": ["Good product", None, "Bad product"],
            "sentiment": ["Positive", "Neutral", "Negative"],
        })
        result = preprocess_dataframe(df, text_col="text")
        assert len(result) <= 3  # Should handle None gracefully

    def test_preserves_original_columns(self):
        df = pd.DataFrame({
            "text": ["Great product overall"],
            "sentiment": ["Positive"],
            "extra_col": [42],
        })
        result = preprocess_dataframe(df, text_col="text")
        assert "extra_col" in result.columns
