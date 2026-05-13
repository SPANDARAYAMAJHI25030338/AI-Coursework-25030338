"""Phase 4a — generate sentence embeddings + FAISS index for all MDCC."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import config
from src import retrieval_pipeline as rp


def main() -> None:
    df = pd.read_csv(config.PROCESSED_CSV)
    print(f"[load] {len(df):,} rows")

    texts = df["text"].astype(str).tolist()
    print(f"[embed] using {config.EMBED_MODEL} on CPU ...")
    embs = rp.embed(texts, batch_size=64, show_progress=True)
    print(f"[embed] shape={embs.shape}  dtype={embs.dtype}")

    config.EMBEDDINGS_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(config.EMBEDDINGS_NPY, embs)
    print(f"[save] {config.EMBEDDINGS_NPY}  ({embs.nbytes/1e6:.1f} MB)")

    print("[index] building FAISS IndexFlatIP ...")
    index = rp.build_index(embs, path=config.FAISS_INDEX)
    print(f"[save] {config.FAISS_INDEX}  ntotal={index.ntotal}")


if __name__ == "__main__":
    main()
