"""
Data Loader Module.

Responsible for loading raw datasets from disk or user uploads,
performing initial validation, and returning clean DataFrames.
"""

from pathlib import Path
from typing import Optional, List

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_csv(
    path: Path,
    column_names: Optional[List[str]] = None,
    has_header: bool = False,
) -> pd.DataFrame:
    """
    Load a CSV file into a Pandas DataFrame.

    Args:
        path:         Path to the CSV file.
        column_names: Optional list of column names to assign.
        has_header:   Whether the CSV contains a header row.

    Returns:
        A validated ``pd.DataFrame``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the resulting DataFrame is empty.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    header = 0 if has_header else None
    df = pd.read_csv(path, header=header, names=column_names, encoding="utf-8", on_bad_lines="skip")

    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    logger.info(
        "Dataset loaded from %s — shape: %s",
        path.name,
        df.shape,
    )
    return df


def load_uploaded_csv(
    uploaded_file,
    column_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load a CSV from a Streamlit ``UploadedFile`` object.

    Args:
        uploaded_file: Streamlit file-upload widget output.
        column_names:  Optional column names.

    Returns:
        A ``pd.DataFrame`` parsed from the upload.
    """
    df = pd.read_csv(uploaded_file, header=None, names=column_names, on_bad_lines="skip")
    df.dropna(inplace=True)

    logger.info(
        "Uploaded dataset parsed — shape: %s",
        df.shape,
    )
    return df
