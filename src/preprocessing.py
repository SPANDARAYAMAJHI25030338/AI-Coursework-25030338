"""Shared preprocessing pipeline.

All three models (TF-IDF, transformer, retrieval) MUST call `clean_text` so the
comparative evaluation is fair.

Educational note: differences in preprocessing across models invalidate the
comparison — a model can look better just because it received cleaner input.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import pandas as pd

from src import config


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_PUNCT_RE = re.compile(r"([!?.])\1{2,}")  # !!! → !!!  (we keep some)


def clean_text(text: str) -> str:
    """Normalise a single piece of campaign text.

    Steps:
      1. Coerce to str (handle NaN gracefully).
      2. NFKC unicode normalisation (collapses stylistic quote/dash/emoji variants).
      3. Replace URLs with <URL> placeholder (presence may itself be a signal).
      4. Collapse runs of whitespace.
      5. Lightly cap absurd punctuation repetition (e.g. 10 exclamations → 3).
    We deliberately do NOT lowercase, remove emotional words, or strip punctuation.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _URL_RE.sub(" <URL> ", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1\1\1", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_series(texts: Iterable[str]) -> pd.Series:
    """Vectorised version of `clean_text` for a pandas Series."""
    return pd.Series([clean_text(t) for t in texts])


def load_mdcc(path=None) -> pd.DataFrame:
    """Load MDCC, apply preprocessing, and return a tidy DataFrame.

    The output always has columns: campaign_id, category, text (cleaned), raised, goal.
    """
    path = path or config.MDCC_CSV
    df = pd.read_csv(path, low_memory=False)
    # MDCC ships with no nulls in clean_description per our earlier audit,
    # but we guard anyway for robustness.
    df = df.dropna(subset=[config.MDCC_TEXT_COL]).copy()
    df["text"] = clean_series(df[config.MDCC_TEXT_COL])
    # Drop ultra-short rows that survived MDCC's own cleaning
    df = df[df["text"].str.len() >= 25].reset_index(drop=True)
    keep = [config.MDCC_ID_COL, config.MDCC_CAT_COL, "text", "goal", "raised"]
    return df[keep].rename(columns={config.MDCC_ID_COL: "campaign_id",
                                    config.MDCC_CAT_COL: "category"})
