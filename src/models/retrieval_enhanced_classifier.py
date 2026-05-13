"""Model 3: retrieval-enhanced classifier.

Idea:
  - For each test example, retrieve k nearest training examples by sentence
    embedding similarity.
  - Combine the transformer's prediction with a "neighbour vote" — the majority
    label among the retrieved neighbours, weighted by similarity score.
  - Final prediction: log-linear combination of transformer probability and
    neighbour-vote probability.

This is intentionally lightweight: it's a *classification* pipeline enhanced
with retrieved evidence, not a full generative RAG system. That matches the
project doc's "lightweight, NOT production-grade RAG" guidance.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src import config


def neighbour_vote(neighbour_idx: np.ndarray, neighbour_sim: np.ndarray,
                   train_labels: Sequence[int], n_classes: int = 2) -> np.ndarray:
    """Convert (n_queries, k) neighbour indices into a probability matrix.

    Each neighbour contributes its label, weighted by cosine similarity.
    """
    train_labels = np.asarray(train_labels)
    n_queries = neighbour_idx.shape[0]
    proba = np.zeros((n_queries, n_classes), dtype="float32")
    for i in range(n_queries):
        lbls = train_labels[neighbour_idx[i]]
        sims = neighbour_sim[i]
        # softmax-normalise similarities to weight neighbours by relevance
        w = np.exp(sims - sims.max())
        w = w / w.sum()
        for lbl, ww in zip(lbls, w):
            proba[i, lbl] += ww
    return proba


def combine(transformer_proba: np.ndarray, retrieval_proba: np.ndarray,
            alpha: float = 0.7) -> np.ndarray:
    """Convex combination: alpha * transformer + (1-alpha) * retrieval."""
    return alpha * transformer_proba + (1.0 - alpha) * retrieval_proba


def predict_combined(transformer_proba: np.ndarray, neighbour_idx: np.ndarray,
                     neighbour_sim: np.ndarray, train_labels: Sequence[int],
                     alpha: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Run the full retrieval-enhanced prediction. Returns (preds, proba)."""
    n_classes = transformer_proba.shape[1]
    retr_proba = neighbour_vote(neighbour_idx, neighbour_sim, train_labels, n_classes)
    combined = combine(transformer_proba, retr_proba, alpha=alpha)
    preds = combined.argmax(axis=1)
    return preds, combined
