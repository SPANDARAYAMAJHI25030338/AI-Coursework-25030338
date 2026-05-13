"""Retrieval pipeline — sentence embeddings + FAISS index.

Educational note: a sentence-transformer maps a sentence to a fixed-size dense
vector where semantically similar sentences are close together. FAISS is a
library for fast nearest-neighbour search in such vector spaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src import config


_model_cache: SentenceTransformer | None = None


def get_embed_model() -> SentenceTransformer:
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer(config.EMBED_MODEL)
    return _model_cache


def embed(texts: Sequence[str], batch_size: int = 64,
          show_progress: bool = True) -> np.ndarray:
    """Encode a list of texts to a (n, d) float32 array, L2-normalised."""
    model = get_embed_model()
    embs = model.encode(list(texts), batch_size=batch_size,
                        show_progress_bar=show_progress,
                        convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype("float32")


def build_index(embeddings: np.ndarray, path: Path | None = None) -> faiss.Index:
    """Build a cosine-similarity FAISS index (inner product on L2-normalised vectors).

    Cosine sim is appropriate for sentence embeddings; normalising first lets us
    use the cheaper Inner-Product index.
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))
    return index


def load_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))


def topk_neighbours(index: faiss.Index, query_emb: np.ndarray, k: int = 5):
    """Return (scores, idx) arrays of shape (n, k)."""
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    return index.search(query_emb, k)
