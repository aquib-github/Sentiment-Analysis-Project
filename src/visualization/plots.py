"""
Visualization Module.

Generates all charts and plots used throughout the Streamlit
application, including distribution charts, word clouds,
confusion matrices, and model comparison charts.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from wordcloud import WordCloud

import config
from src.utils.logger import get_logger

matplotlib.use("Agg")  # Non-interactive backend for Streamlit
logger = get_logger(__name__)


def plot_sentiment_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Create a bar + pie chart showing sentiment class distribution.

    Args:
        df: DataFrame with a ``sentiment`` column.

    Returns:
        Matplotlib Figure.
    """
    counts = df[config.TARGET_COLUMN].value_counts()
    colors = [config.SENTIMENT_COLORS.get(s, "#95a5a6") for s in counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sentiment Distribution", fontsize=16, fontweight="bold")

    # Bar chart
    bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor="black")
    axes[0].set_title("Count per Sentiment", fontsize=13)
    axes[0].set_xlabel("Sentiment")
    axes[0].set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, val + max(counts.values) * 0.02,
            str(val), ha="center", fontweight="bold", fontsize=10,
        )

    # Pie chart
    axes[1].pie(
        counts.values, labels=counts.index, autopct="%1.1f%%",
        colors=colors, startangle=140, textprops={"fontsize": 10},
    )
    axes[1].set_title("Sentiment Share", fontsize=13)

    plt.tight_layout()
    return fig


def plot_wordclouds(df: pd.DataFrame) -> plt.Figure:
    """
    Generate a 2×2 grid of word clouds, one per sentiment class.

    Args:
        df: DataFrame with ``sentiment`` and ``clean_text`` columns.

    Returns:
        Matplotlib Figure.
    """
    sentiments = df[config.TARGET_COLUMN].unique()[:4]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes = axes.flatten()

    for i, sentiment in enumerate(sentiments):
        text_blob = " ".join(df[df[config.TARGET_COLUMN] == sentiment]["clean_text"])
        if not text_blob.strip():
            axes[i].text(0.5, 0.5, "No data", ha="center", va="center")
            axes[i].set_title(f"{sentiment} Tweets", fontsize=13, fontweight="bold")
            axes[i].axis("off")
            continue

        cmap = config.WORDCLOUD_CMAPS.get(sentiment, "viridis")
        wc = WordCloud(
            width=600, height=320, background_color="white",
            colormap=cmap, max_words=100,
        ).generate(text_blob)

        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(f"{sentiment} Tweets", fontsize=13, fontweight="bold")
        axes[i].axis("off")

    # Hide unused subplots
    for j in range(len(sentiments), 4):
        axes[j].axis("off")

    fig.suptitle("Word Clouds by Sentiment", fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: List[str],
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Plot a single confusion matrix heatmap.

    Args:
        cm:          Confusion matrix array.
        label_names: Class names for axis labels.
        title:       Chart title.
        cmap:        Colormap name.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap, ax=ax,
        xticklabels=label_names, yticklabels=label_names,
        linewidths=0.5, linecolor="white",
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    return fig


def plot_all_confusion_matrices(
    evaluations: Dict[str, dict],
    label_names: List[str],
) -> plt.Figure:
    """
    Plot confusion matrices for all models side by side.

    Args:
        evaluations: Dict mapping model name → metrics dict.
        label_names: Class names for axis labels.

    Returns:
        Matplotlib Figure.
    """
    model_names = list(evaluations.keys())
    n = len(model_names)
    cmaps = ["Purples", "Greens", "Blues", "Oranges"]

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    fig.suptitle("Confusion Matrices", fontsize=16, fontweight="bold")

    for ax, name, cmap in zip(axes, model_names, cmaps[:n]):
        cm = evaluations[name]["confusion_matrix"]
        acc = evaluations[name]["accuracy"] * 100
        sns.heatmap(
            cm, annot=True, fmt="d", cmap=cmap, ax=ax,
            xticklabels=label_names, yticklabels=label_names,
            linewidths=0.5, linecolor="white",
        )
        ax.set_title(f"{name}\n{acc:.2f}% accuracy", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    return fig


def plot_model_comparison(evaluations: Dict[str, dict]) -> plt.Figure:
    """
    Create accuracy and F1 bar charts comparing all models.

    Args:
        evaluations: Dict mapping model name → metrics dict.

    Returns:
        Matplotlib Figure.
    """
    names = list(evaluations.keys())
    accs = [evaluations[n]["accuracy"] * 100 for n in names]
    f1s = [evaluations[n]["f1"] for n in names]
    bar_colors = ["#9b59b6", "#3498db", "#2ecc71", "#e67e22"][:len(names)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Comparison", fontsize=16, fontweight="bold")

    for ax, values, title, fmt, ylim in zip(
        axes,
        [accs, f1s],
        ["Accuracy (%)", "F1 Score"],
        ["{:.1f}%", "{:.4f}"],
        [105, 1.1],
    ):
        bars = ax.bar(names, values, color=bar_colors, edgecolor="black", width=0.5)
        ax.set_title(title, fontsize=13)
        ax.set_ylim(0, ylim)
        ax.tick_params(axis="x", rotation=10)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ylim * 0.01,
                fmt.format(val),
                ha="center", fontweight="bold", fontsize=10,
            )

    plt.tight_layout()
    return fig
