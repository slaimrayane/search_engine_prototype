import hashlib
import os
import re
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
FALLBACK_DIMENSION = 384


@lru_cache(maxsize=1)
def _load_model():
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)


def _token_hash_embedding(text, dimension=FALLBACK_DIMENSION):
    """Deterministic local fallback used when the embedding model is not cached."""
    vector = np.zeros(dimension, dtype="float32")
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{1,}", text.lower())
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = np.linalg.norm(vector)
    if norm:
        vector /= norm
    return vector


def embed_text(text):
    try:
        return _load_model().encode(text, convert_to_numpy=True).astype("float32")
    except Exception:
        return _token_hash_embedding(text)


def embed_documents(documents):
    return np.array([embed_text(text) for _, text in documents], dtype="float32")
