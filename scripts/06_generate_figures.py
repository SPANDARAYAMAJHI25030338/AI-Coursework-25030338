"""Phase 5b — generate all figures for the final report.

Reads metrics + predictions from outputs/results/ and emits PNGs to
outputs/figures/. Each figure is 300 DPI and report-ready.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import config
from src import visualizations as viz


def _load_metrics(name: str) -> dict | None:
    p = config.RESULTS_DIR / f"{name}_metrics.json"
    if not p.exists():
        print(f"[skip] {p.name} not found")
        return None
    return json.loads(p.read_text())


def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.PROCESSED_CSV)

    print("[fig] class distribution")
    viz.plot_class_distribution(df["binary_label"].tolist())

    print("[fig] text length histogram")
    viz.plot_text_length_histogram(df["text"].astype(str).str.len().tolist())

    print("[fig] system architecture")
    viz.plot_system_architecture()

    # Confusion matrices + comparison
    metric_rows = []
    for model_name, file_stem in [
        ("TF-IDF + LR", "tfidf_lr"),
        ("DistilBERT", "distilbert"),
        ("Retrieval-Enhanced", "retrieval_enhanced"),
    ]:
        m = _load_metrics(file_stem)
        if m is None:
            continue
        metric_rows.append(m)
        cm = np.asarray(m["confusion"])
        print(f"[fig] confusion {model_name}")
        viz.plot_confusion_matrix(cm, labels=config.LABELS_BINARY,
                                  model_name=model_name)

    if metric_rows:
        print("[fig] model comparison + F1")
        viz.plot_model_comparison(metric_rows)
        viz.plot_f1_comparison(metric_rows)

    # Training-loss curve for the transformer (if history.json exists)
    hist_path = config.RESULTS_DIR / "distilbert_history.json"
    if hist_path.exists():
        print("[fig] training loss curve")
        hist = json.loads(hist_path.read_text())
        import matplotlib.pyplot as plt
        epochs = [h["epoch"] for h in hist]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, [h["train_loss"] for h in hist], "o-", label="train loss")
        ax.plot(epochs, [h["val_loss"] for h in hist], "s-", label="val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy")
        ax.set_title("DistilBERT training curve")
        ax.legend()
        fig.savefig(config.FIGURES_DIR / "fig_training_loss.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Alpha ablation (if present)
    alpha_path = config.RESULTS_DIR / "retrieval_alpha_ablation.csv"
    if alpha_path.exists():
        print("[fig] retrieval alpha ablation")
        ab = pd.read_csv(alpha_path)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ab["alpha"], ab["accuracy"], "o-", label="accuracy")
        ax.plot(ab["alpha"], ab["f1"], "s-", label="F1")
        ax.plot(ab["alpha"], ab["macro_f1"], "^-", label="macro F1")
        ax.set_xlabel("alpha (transformer weight)")
        ax.set_ylabel("score")
        ax.set_title("Retrieval-enhanced classifier — alpha ablation")
        ax.legend()
        fig.savefig(config.FIGURES_DIR / "fig_alpha_ablation.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"[done] figures in {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
