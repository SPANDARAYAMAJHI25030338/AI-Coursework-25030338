"""Auxiliary feature helpers.

The TF-IDF baseline uses sklearn's TfidfVectorizer directly inside its Pipeline.
This module provides additional engineered features (text length, punctuation
density, emotion-cue density) that can optionally be concatenated for ablation
experiments. Kept simple — not used by the core pipeline by default.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


PUNCT_RE = re.compile(r"[!?]")
ALLCAPS_RE = re.compile(r"\b[A-Z]{3,}\b")


def length_feature(texts: Iterable[str]) -> np.ndarray:
    return np.array([len(t) for t in texts]).reshape(-1, 1)


def punctuation_density(texts: Iterable[str]) -> np.ndarray:
    out = []
    for t in texts:
        if not t:
            out.append(0.0)
            continue
        out.append(len(PUNCT_RE.findall(t)) / max(1, len(t.split())))
    return np.array(out).reshape(-1, 1)


def allcaps_density(texts: Iterable[str]) -> np.ndarray:
    out = []
    for t in texts:
        if not t:
            out.append(0.0)
            continue
        out.append(len(ALLCAPS_RE.findall(t)) / max(1, len(t.split())))
    return np.array(out).reshape(-1, 1)


def engineered_matrix(texts: Iterable[str]) -> np.ndarray:
    """Stack the engineered features into one (n, 3) matrix."""
    return np.hstack([
        length_feature(texts),
        punctuation_density(texts),
        allcaps_density(texts),
    ]).astype("float32")
