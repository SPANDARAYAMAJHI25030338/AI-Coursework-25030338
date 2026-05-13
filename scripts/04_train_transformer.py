"""Phase 3 — fine-tune DistilBERT on MDCC + weak labels.

On CPU we subsample to TRANSFORMER_TRAIN_SUBSAMPLE rows for tractability. The
choice is documented in config.py and acknowledged in the report's Limitations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.evaluation import compute_metrics
from src.models import transformer_classifier as tc


def main() -> None:
    df = pd.read_csv(config.PROCESSED_CSV)
    df["y"] = df["binary_label"].map(config.LABEL2ID)
    print(f"[load] {len(df):,} rows")

    # Stratified subsample to keep CPU training tractable
    if len(df) > config.TRANSFORMER_TRAIN_SUBSAMPLE:
        df, _ = train_test_split(
            df,
            train_size=config.TRANSFORMER_TRAIN_SUBSAMPLE,
            stratify=df["y"],
            random_state=config.RANDOM_SEED,
        )
        df = df.reset_index(drop=True)
        print(f"[subsample] kept {len(df):,} rows for transformer training")

    X = df["text"].astype(str).tolist()
    y = df["y"].tolist()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TRAIN_TEST_SPLIT,
        stratify=y, random_state=config.RANDOM_SEED,
    )
    # Further split a small val set off the train side
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_tr, y_tr, test_size=0.10,
        stratify=y_tr, random_state=config.RANDOM_SEED,
    )
    print(f"[split] train={len(X_tr2):,}  val={len(X_val):,}  test={len(X_te):,}")

    print(f"[train] fine-tuning {config.TRANSFORMER_NAME} for "
          f"{config.TRANSFORMER_EPOCHS} epochs on CPU ...")
    save_dir = config.MODELS_DIR / "distilbert"
    model, tok, history = tc.fine_tune(
        X_tr2, y_tr2, X_val, y_val, save_dir=save_dir,
    )

    print("[eval] running on test split ...")
    y_pred, y_proba = tc.predict(model, tok, X_te,
                                  batch_size=config.TRANSFORMER_EVAL_BATCH)

    m = compute_metrics(np.array(y_te), y_pred, y_proba, "DistilBERT",
                        labels=config.LABELS_BINARY)
    print(f"[metrics] acc={m.accuracy:.4f}  f1={m.f1:.4f}  "
          f"macro_f1={m.macro_f1:.4f}  roc_auc={m.roc_auc}")
    print(m.per_class_report)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "distilbert_metrics.json").write_text(
        json.dumps(m.to_dict(), indent=2))
    (config.RESULTS_DIR / "distilbert_history.json").write_text(
        json.dumps(history, indent=2))
    np.savez(config.RESULTS_DIR / "distilbert_preds.npz",
             y_true=np.array(y_te), y_pred=y_pred, y_proba=y_proba,
             X_test=np.array(X_te, dtype=object))
    print("[save] distilbert metrics, history, predictions written")


if __name__ == "__main__":
    main()
