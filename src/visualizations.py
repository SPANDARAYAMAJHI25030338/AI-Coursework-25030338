"""Plotting helpers.

All figures are saved at 300 DPI to outputs/figures/ for direct inclusion in the
final report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config


sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 120


def _save(fig, name: str) -> Path:
    out = config.FIGURES_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_class_distribution(labels: Sequence[str], save_as: str = "fig_class_distribution.png") -> Path:
    counts = pd.Series(labels).value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="muted")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of campaigns")
    ax.set_xlabel("Label")
    ax.set_title("MDCC weak-label class distribution")
    return _save(fig, save_as)


def plot_text_length_histogram(lengths: Sequence[int], save_as: str = "fig_text_length.png") -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(lengths, bins=60, ax=ax, color="steelblue")
    ax.axvline(np.median(lengths), color="red", linestyle="--",
               label=f"median = {int(np.median(lengths))}")
    ax.set_xlabel("Characters per campaign description")
    ax.set_ylabel("Count")
    ax.set_title("MDCC campaign description length distribution")
    ax.legend()
    return _save(fig, save_as)


def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], model_name: str,
                          save_as: str | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {model_name}")
    name = save_as or f"fig_confusion_{model_name.lower().replace(' ', '_')}.png"
    return _save(fig, name)


def plot_model_comparison(metrics_rows: list[dict], save_as: str = "fig_model_comparison.png") -> Path:
    df = pd.DataFrame(metrics_rows)
    metric_cols = ["accuracy", "precision", "recall", "f1", "macro_f1"]
    melted = df.melt(id_vars="model", value_vars=metric_cols,
                     var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=melted, x="metric", y="score", hue="model", ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_title("Model comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.legend(title="Model", loc="lower right")
    return _save(fig, save_as)


def plot_f1_comparison(metrics_rows: list[dict], save_as: str = "fig_f1_comparison.png") -> Path:
    df = pd.DataFrame(metrics_rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    order = df.sort_values("f1", ascending=False)
    sns.barplot(data=order, x="model", y="f1", ax=ax, palette="crest")
    for i, v in enumerate(order["f1"].values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 score")
    ax.set_xlabel("")
    ax.set_title("F1 score across models")
    return _save(fig, save_as)


def plot_system_architecture(save_as: str = "fig_system_architecture.png") -> Path:
    """Hand-drawn ASCII-style architecture diagram rendered with matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    boxes = [
        (0.02, 0.55, 0.18, 0.30, "MDCC GoFundMe\n(14,961 campaigns)\nclean_description"),
        (0.02, 0.10, 0.18, 0.30, "Weak labels\n(keyword + emotion)"),
        (0.25, 0.55, 0.18, 0.30, "Preprocessing\nNFKC + URL + ws"),
        (0.48, 0.78, 0.18, 0.15, "TF-IDF + LR\n(baseline)"),
        (0.48, 0.55, 0.18, 0.15, "DistilBERT\n(transformer)"),
        (0.48, 0.32, 0.18, 0.15, "DistilBERT + FAISS\n(retrieval-enhanced)"),
        (0.74, 0.55, 0.18, 0.30, "Comparative\nevaluation\n(metrics + figs)"),
    ]
    for (x, y, w, h, txt) in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor="black",
                                    facecolor="#e6f0fa", lw=1.2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)

    arrows = [
        (0.20, 0.70, 0.05, 0.0),   # MDCC -> Preprocessing
        (0.20, 0.25, 0.05, 0.20),  # Weak labels -> Preprocessing area
        (0.43, 0.70, 0.05, 0.15),  # Preproc -> TF-IDF
        (0.43, 0.62, 0.05, 0.00),  # Preproc -> DistilBERT
        (0.43, 0.55, 0.05, -0.15), # Preproc -> Retrieval
        (0.66, 0.85, 0.08, -0.15), # TF-IDF -> Eval
        (0.66, 0.62, 0.08, 0.08),  # DistilBERT -> Eval
        (0.66, 0.39, 0.08, 0.30),  # Retrieval -> Eval
    ]
    for (x, y, dx, dy) in arrows:
        ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", lw=1.0))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Crowdfunding manipulation detection — system architecture")
    return _save(fig, save_as)
