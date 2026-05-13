"""Model 1: TF-IDF + Logistic Regression baseline.

Educational note: TF-IDF (term frequency × inverse document frequency) gives each
word a weight that's high when it appears often in this document but rarely
across the whole corpus. Logistic Regression then learns a linear boundary in
that high-dimensional sparse space. It's a strong, fast, interpretable baseline.
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config


def build_pipeline() -> Pipeline:
    """Construct the standard TF-IDF + Logistic Regression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            sublinear_tf=config.TFIDF_SUBLINEAR_TF,
            strip_accents="unicode",
            lowercase=True,
            min_df=2,
            max_df=0.95,
        )),
        ("lr", LogisticRegression(
            max_iter=config.LR_MAX_ITER,
            class_weight=config.LR_CLASS_WEIGHT,
            n_jobs=-1,
            random_state=config.RANDOM_SEED,
        )),
    ])


def fit_predict(X_train, y_train, X_test):
    """Fit the pipeline and return (pipeline, y_pred, y_proba)."""
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)
    return pipe, y_pred, y_proba


def save(pipe, path=None):
    path = path or (config.MODELS_DIR / "tfidf_lr.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    return path
