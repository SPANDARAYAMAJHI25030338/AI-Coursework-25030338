"""Phase 4b — retrieval-enhanced classifier.

Uses the saved FAISS index and the saved DistilBERT predictions.
Combines transformer probability with neighbour-vote probability.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss
import numpy as np
import pandas as pd

from src import config
from src.evaluation import compute_metrics
from src.models import retrieval_enhanced_classifier as rec
from src import retrieval_pipeline as rp


def main() -> None:
    print("[load] processed data + weak labels")
    df = pd.read_csv(config.PROCESSED_CSV)
    df["y"] = df["binary_label"].map(config.LABEL2ID)

    print(f"[load] embeddings <- {config.EMBEDDINGS_NPY}")
    all_emb = np.load(config.EMBEDDINGS_NPY)
    assert len(all_emb) == len(df), "embeddings and processed CSV must align"

    print(f"[load] transformer test predictions <- distilbert_preds.npz")
    bert_preds = np.load(config.RESULTS_DIR / "distilbert_preds.npz",
                         allow_pickle=True)
    test_texts = bert_preds["X_test"].tolist()
    y_true = bert_preds["y_true"]
    bert_proba = bert_preds["y_proba"]

    # Find indices of test campaigns in the full df (match by text)
    text_to_idx = {t: i for i, t in enumerate(df["text"].astype(str).tolist())}
    test_idx = np.array([text_to_idx[t] for t in test_texts])
    train_mask = np.ones(len(df), dtype=bool)
    train_mask[test_idx] = False
    train_idx = np.where(train_mask)[0]

    print(f"[index] building training-only FAISS index (n={len(train_idx):,})")
    train_emb = all_emb[train_idx]
    test_emb = all_emb[test_idx]
    train_labels = df["y"].values[train_idx]
    index = rp.build_index(train_emb)

    k = config.RETRIEVAL_TOP_K
    print(f"[retrieve] top-{k} neighbours per test query ...")
    sims, neigh = rp.topk_neighbours(index, test_emb, k=k)

    print(f"[combine] alpha=0.7 (0.7 transformer + 0.3 retrieval)")
    y_pred, y_proba = rec.predict_combined(
        bert_proba, neigh, sims, train_labels, alpha=0.7)

    m = compute_metrics(y_true, y_pred, y_proba, "Retrieval-Enhanced",
                        labels=config.LABELS_BINARY)
    print(f"[metrics] acc={m.accuracy:.4f}  f1={m.f1:.4f}  "
          f"macro_f1={m.macro_f1:.4f}  roc_auc={m.roc_auc}")
    print(m.per_class_report)

    (config.RESULTS_DIR / "retrieval_enhanced_metrics.json").write_text(
        json.dumps(m.to_dict(), indent=2))
    np.savez(config.RESULTS_DIR / "retrieval_enhanced_preds.npz",
             y_true=y_true, y_pred=y_pred, y_proba=y_proba,
             neighbour_idx=neigh, neighbour_sim=sims)
    print("[save] retrieval_enhanced metrics + predictions written")

    # Bonus: ablation across alpha
    print("\n[ablation] alpha sweep")
    rows = []
    for a in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
        yp, _ = rec.predict_combined(bert_proba, neigh, sims, train_labels, alpha=a)
        mm = compute_metrics(y_true, yp, None, f"alpha={a:.1f}",
                             labels=config.LABELS_BINARY)
        rows.append({"alpha": a, "accuracy": mm.accuracy, "f1": mm.f1,
                     "macro_f1": mm.macro_f1})
        print(f"  alpha={a:.1f}  acc={mm.accuracy:.4f}  f1={mm.f1:.4f}")
    pd.DataFrame(rows).to_csv(
        config.RESULTS_DIR / "retrieval_alpha_ablation.csv", index=False)


if __name__ == "__main__":
    main()
