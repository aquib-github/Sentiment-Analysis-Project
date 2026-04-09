"""
Data Preprocessing Module.

Handles all text-cleaning and NLP preprocessing steps including
lowercasing, URL/mention removal, punctuation stripping,
stop-word filtering, and lemmatization.
"""

import re
import string
from typing import List

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Download NLTK resources (idempotent) ─────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Apply a full NLP cleaning pipeline to a single text string.

    Steps:
        1. Lowercase
        2. Remove URLs, mentions, hashtags, numbers
        3. Remove punctuation
        4. Tokenize
        5. Remove stop-words and short tokens (len ≤ 2)
        6. Lemmatize

    Args:
        text: Raw input text.

    Returns:
        Cleaned and lemmatized text string.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|@\w+|#\w+|\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens: List[str] = text.split()
    tokens = [
        _LEMMATIZER.lemmatize(w)
        for w in tokens
        if w not in _STOP_WORDS and len(w) > 2
    ]
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Clean the text column of a DataFrame and remove empty rows.

    Args:
        df:       Input DataFrame (must contain ``text_col``).
        text_col: Name of the column containing raw text.

    Returns:
        DataFrame with a new ``clean_text`` column.
    """
    logger.info("Starting text preprocessing on %d rows …", len(df))

    df = df.copy()
    df.dropna(subset=[text_col], inplace=True)
    df[text_col] = df[text_col].astype(str)

    df["clean_text"] = df[text_col].apply(clean_text)
    df = df[df["clean_text"].str.strip() != ""]

    logger.info("Preprocessing complete — %d rows remaining.", len(df))
    return df
