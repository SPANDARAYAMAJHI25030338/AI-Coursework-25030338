"""Phase 2 — TF-IDF + Logistic Regression baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.evaluation import compute_metrics
from src.models import traditional_ml
from src.explainability import top_features_per_class


def main() -> None:
    df = pd.read_csv(config.PROCESSED_CSV)
    print(f"[load] {len(df):,} rows from {config.PROCESSED_CSV}")

    df["y"] = df["binary_label"].map(config.LABEL2ID)
    X = np.asarray(df["text"].astype(str).tolist(), dtype=object)
    y = np.asarray(df["y"].tolist(), dtype=np.int64)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TRAIN_TEST_SPLIT, stratify=y,
        random_state=config.RANDOM_SEED,
    )
    print(f"[split] train={len(X_tr):,}  test={len(X_te):,}")

    print("[train] TF-IDF + LogisticRegression ...")
    pipe, y_pred, y_proba = traditional_ml.fit_predict(X_tr, y_tr, X_te)

    m = compute_metrics(y_te, y_pred, y_proba, "TF-IDF + LR",
                        labels=config.LABELS_BINARY)
    print(f"[metrics] acc={m.accuracy:.4f}  f1={m.f1:.4f}  "
          f"macro_f1={m.macro_f1:.4f}  roc_auc={m.roc_auc}")
    print(m.per_class_report)

    # Save artefacts
    model_path = traditional_ml.save(pipe)
    print(f"[save] model -> {model_path}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "tfidf_lr_metrics.json"
    out.write_text(json.dumps(m.to_dict(), indent=2))
    print(f"[save] metrics -> {out}")

    # Save predictions for later combination with retrieval
    np.savez(config.RESULTS_DIR / "tfidf_lr_preds.npz",
             y_true=y_te, y_pred=y_pred, y_proba=y_proba)

    # Save top features for explainability section
    feats = top_features_per_class(pipe, n=25)
    feats_path = config.RESULTS_DIR / "tfidf_lr_top_features.csv"
    feats.to_csv(feats_path, index=False)
    print(f"[save] top features -> {feats_path}")


if __name__ == "__main__":
    main()
