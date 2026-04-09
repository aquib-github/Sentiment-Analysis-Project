"""
Helper Utilities Module.

Contains reusable helper functions used across the project,
such as file I/O wrappers, timing decorators, and common checks.
"""

import time
import functools
from pathlib import Path
from typing import Any

import joblib

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_artifact(obj: Any, path: Path) -> None:
    """
    Persist a Python object to disk using joblib.

    Args:
        obj:  The object to serialize.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Artifact saved → %s", path)


def load_artifact(path: Path) -> Any:
    """
    Load a previously saved joblib artifact from disk.

    Args:
        path: Path to the serialized file.

    Returns:
        The deserialized Python object.

    Raises:
        FileNotFoundError: If the artifact file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    obj = joblib.load(path)
    logger.info("Artifact loaded ← %s", path)
    return obj


def timer(func):
    """Decorator that logs the execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%s executed in %.2f seconds", func.__name__, elapsed)
        return result

    return wrapper


def file_exists(path: Path) -> bool:
    """Check whether a file exists at the given path."""
    return path.is_file()
