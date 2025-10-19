import os
from typing import List, Optional
import numpy as np

# try to load sentence-transformers safely
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

_model = None

def get_model(model_name: str = "all-MiniLM-L6-v2"):
    global _model
    if _model is None:
        if not _HAS_ST:
            raise RuntimeError("sentence-transformers not available. Install sentence-transformers or fallback to TF-IDF.")
        _model = SentenceTransformer(model_name)
    return _model

def embed_text(texts: List[str]) -> List[List[float]]:
    """
    Returns list of vectors. If sentence-transformers is not available,
    raise an error — caller can fall back to TF-IDF.
    """
    model = get_model()
    vecs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vecs.tolist()

def normalize_vector(vec):
    a = np.array(vec, dtype=float)
    norm = np.linalg.norm(a)
    if norm == 0:
        return a.tolist()
    return (a / norm).tolist()
