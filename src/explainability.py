"""Light-weight explainability: feature-coefficient inspection for TF-IDF + LR
and keyword-attribution summary for the transformer.

A full SHAP / attention visualisation is computationally expensive for long MDCC
texts on CPU. We provide simple, defensible alternatives here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def top_features_per_class(pipeline, n: int = 25) -> pd.DataFrame:
    """Return the top-n positive- and negative-weight features for the LR model."""
    tfidf = pipeline.named_steps["tfidf"]
    lr = pipeline.named_steps["lr"]
    feature_names = np.asarray(tfidf.get_feature_names_out())
    coefs = lr.coef_.ravel()  # binary -> single coefficient vector
    top_pos = np.argsort(coefs)[-n:][::-1]
    top_neg = np.argsort(coefs)[:n]
    return pd.DataFrame({
        "manipulative_top": feature_names[top_pos],
        "manipulative_weight": coefs[top_pos].round(4),
        "non_manipulative_top": feature_names[top_neg],
        "non_manipulative_weight": coefs[top_neg].round(4),
    })


def attribution_by_lexicon(texts, lexicon_scores: pd.DataFrame) -> pd.DataFrame:
    """Summarise which weak-label lexicon category dominates each prediction."""
    cols = [c for c in lexicon_scores.columns if c.startswith("score_")]
    summary = lexicon_scores[cols].copy()
    summary["dominant"] = summary[cols].idxmax(axis=1).str.replace("score_", "", regex=False)
    return summary
