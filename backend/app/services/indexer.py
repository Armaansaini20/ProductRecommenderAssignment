# app/services/indexer.py
from pathlib import Path
import os
import numpy as np
import json
from typing import List, Tuple, Optional

# Try to import Pinecone; only required if USE_PINECONE=true
try:
    import pinecone
except Exception:
    pinecone = None


class VectorIndex:
    """
    Local numpy-backed index. Loads index.npz (ids + matrix) and metadata json.
    matrix rows are normalized on load.
    """
    def __init__(self):
        self.ids: List[str] = []
        self.matrix: Optional[np.ndarray] = None
        self.metadata: dict = {}

    def load(self, npz_path: str, meta_json_path: Optional[str] = None):
        p = Path(npz_path)
        if not p.exists():
            raise FileNotFoundError(f"Index file not found: {npz_path}")
        data = np.load(str(p), allow_pickle=True)
        # ensure ids as strings
        self.ids = [str(x) for x in data["ids"].tolist()]
        self.matrix = data["matrix"].astype(np.float32)
        # load metadata
        if meta_json_path:
            mpath = Path(meta_json_path)
        else:
            mpath = p.parent / (p.name + ".meta.json")
        if mpath.exists():
            with open(mpath, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        # normalize rows
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = (self.matrix / norms).astype(np.float32)
        return {"n_items": len(self.ids), "dim": self.matrix.shape[1]}

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Returns list of tuples (id, score). Scores are cosine similarity.
        """
        if self.matrix is None:
            return []
        q = query_vector.astype(np.float32)
        qnorm = np.linalg.norm(q)
        if qnorm == 0:
            return []
        q = q / qnorm
        sims = (self.matrix @ q).reshape(-1)
        topk_idx = np.argsort(-sims)[:top_k]
        return [(self.ids[int(i)], float(sims[int(i)])) for i in topk_idx]


class PineconeIndex:
    """
    Wrapper for pinecone Index. Returns list of (id, score, metadata).
    """
    def __init__(self):
        self.index = None
        self.name = os.getenv("PINECONE_INDEX_NAME", "ikarus-products")
        self.initialized = False

    def init(self):
        api_key = os.getenv("PINECONE_API_KEY")
        env = os.getenv("PINECONE_ENV")
        if not api_key or not env:
            raise RuntimeError("Pinecone env vars not set (PINECONE_API_KEY, PINECONE_ENV)")
        if pinecone is None:
            raise RuntimeError("pinecone-client not installed")
        # initialize pinecone
        pinecone.init(api_key=api_key, environment=env)
        if self.name not in pinecone.list_indexes():
            raise RuntimeError(f"Pinecone index '{self.name}' not found. Run upsert script to create it.")
        self.index = pinecone.Index(self.name)
        self.initialized = True

    def search(self, query_vector, top_k=10, filter: Optional[dict] = None):
        if not self.initialized:
            self.init()
        q = query_vector.tolist()
        # include_metadata=True so we get metadata back
        resp = self.index.query(vector=q, top_k=top_k, include_values=False, include_metadata=True, filter=filter)
        matches = resp.get("matches", [])
        results = []
        for m in matches:
            pid = m["id"]
            score = float(m.get("score", 0.0))
            meta = m.get("metadata", {})
            results.append((pid, score, meta))
        return results


# Singleton factory
_index_impl = None


def get_index():
    """
    Returns a singleton instance of either PineconeIndex or VectorIndex
    depending on USE_PINECONE environment variable.
    """
    global _index_impl
    if _index_impl is not None:
        return _index_impl

    use_pc = os.getenv("USE_PINECONE", "false").lower() in ("1", "true", "yes")
    if use_pc:
        _index_impl = PineconeIndex()
    else:
        _index_impl = VectorIndex()
    return _index_impl
