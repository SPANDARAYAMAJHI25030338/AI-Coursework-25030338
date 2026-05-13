"""Phase-1 deliverable: produce data/annotations/mdcc_weak_labels.csv

Run with: python -m scripts.01_generate_weak_labels
(or: python scripts/01_generate_weak_labels.py from the project root)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from src.preprocessing import load_mdcc
from src.weak_labels import label_dataframe


def main() -> None:
    print(f"[load] {config.MDCC_CSV}")
    df = load_mdcc()
    print(f"  rows={len(df):,}  cols={list(df.columns)}")

    print("[label] applying keyword + density heuristic ...")
    labelled = label_dataframe(df)

    binary_counts = labelled["binary_label"].value_counts()
    fine_counts = labelled["fine_label"].value_counts()
    print("[stats] binary_label distribution:")
    print(binary_counts.to_string())
    print(f"  manipulative rate = {binary_counts.get('manipulative', 0)/len(labelled):.3f}")
    print("[stats] fine_label distribution:")
    print(fine_counts.to_string())

    out_path = config.WEAK_LABELS_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labelled.to_csv(out_path, index=False)
    print(f"[save] {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")

    # Persist a small processed version for downstream training scripts
    config.PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    labelled[["campaign_id", "category", "text", "binary_label",
              "fine_label"]].to_csv(config.PROCESSED_CSV, index=False)
    print(f"[save] {config.PROCESSED_CSV}")


if __name__ == "__main__":
    main()
