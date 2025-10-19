# app/routes/recommend.py
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
import numpy as np
import re

from app.models import RecommendRequest, RecommendResponse, RecommendResponseItem
from app.services.clip_embed import embed_texts
from app.services.indexer import get_index

router = APIRouter()

# ---------------------------
# Config / tuning params
# ---------------------------
EXACT_BOOST = 0.30         # large boost when query token appears in title/desc/categories
CATEGORY_BOOST = 0.12
BRAND_BOOST = 0.08
MMR_LAMBDA = 0.75
M_PRESELECT = 100         # number of candidates to preselect before MMR (increase if you want larger candidate pool)

# ---------------------------
# Helpers
# ---------------------------
def l2_normalize_rows_np(a: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return a / norms

def cosine_similarities(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    return (query_vec @ matrix.T).ravel()

def text_exact_match_boost(query: str, ids: List[str], metadata: Dict[str, dict]) -> np.ndarray:
    """
    Return an array of boosts (float) aligned with ids.
    Boost is high if query tokens appear in title/description/categories for that item.
    """
    q_terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 1]
    boosts = np.zeros((len(ids),), dtype=np.float32)
    if not q_terms:
        return boosts
    for i, pid in enumerate(ids):
        meta = metadata.get(pid, {}) or {}
        title = str(meta.get("title") or "").lower()
        desc = str(meta.get("description") or "").lower()
        cats = meta.get("categories") or meta.get("category") or ""
        if isinstance(cats, (list, tuple)):
            cats_str = " ".join(map(str, cats)).lower()
        else:
            cats_str = str(cats).lower()
        # If any query term appears in the title -> big bump
        title_hits = sum(1 for t in q_terms if t in title)
        desc_hits = sum(1 for t in q_terms if t in desc)
        cat_hits = sum(1 for t in q_terms if t in cats_str)
        # weight title more
        score = float(title_hits) * 1.0 + float(desc_hits) * 0.4 + float(cat_hits) * 0.7
        if score > 0:
            # scale by presence; sufficient presence becomes strong boost
            boosts[i] = min(EXACT_BOOST, 0.06 * score + 0.02)
            # small extra if appears in title != matched token multiple times
            if title_hits > 0:
                boosts[i] = min(EXACT_BOOST, boosts[i] + 0.08)
    return boosts

def boost_by_meta(scores: np.ndarray, ids: List[str], metadata: Dict[str, dict],
                  query_meta: Optional[dict], category_boost: float = CATEGORY_BOOST, brand_boost: float = BRAND_BOOST) -> np.ndarray:
    if not query_meta:
        return scores
    cat_q = None
    brand_q = None
    try:
        cat_q = (query_meta.get("categories") or query_meta.get("category") or "")
        if isinstance(cat_q, (list, tuple)):
            cat_q = str(cat_q[0]).lower() if cat_q else ""
        else:
            cat_q = str(cat_q).lower()
    except Exception:
        cat_q = None
    try:
        brand_q = (query_meta.get("brand") or query_meta.get("manufacturer") or "")
        brand_q = str(brand_q).lower() if brand_q else None
    except Exception:
        brand_q = None

    boosted = scores.copy()
    for i, pid in enumerate(ids):
        meta = metadata.get(pid) or {}
        cat = meta.get("categories") or meta.get("category") or ""
        if isinstance(cat, (list, tuple)):
            cat_val = ", ".join(map(str, cat[:3])).lower() if cat else ""
        else:
            cat_val = str(cat).lower() if cat else ""
        if cat_q and cat_val and cat_q in cat_val:
            boosted[i] += category_boost
        b = meta.get("brand") or meta.get("manufacturer") or ""
        b = str(b).lower() if b else ""
        if brand_q and b and brand_q in b:
            boosted[i] += brand_boost
    return boosted

def mmr_rerank(query_vec: np.ndarray, candidate_ids: List[str], candidate_vecs: np.ndarray,
               top_k: int = 10, lambda_param: float = MMR_LAMBDA) -> List[str]:
    if candidate_vecs.shape[0] == 0:
        return []
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    cand = candidate_vecs.copy()
    cand = cand / (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-12)
    sim_to_query = (cand @ q).ravel()
    selected = []
    selected_idx = []

    first = int(np.argmax(sim_to_query))
    selected_idx.append(first)
    selected.append(candidate_ids[first])

    pairwise = cand @ cand.T

    while len(selected) < top_k and len(selected) < len(candidate_ids):
        mmr_scores = []
        for i in range(len(candidate_ids)):
            if i in selected_idx:
                mmr_scores.append(-1e9)
                continue
            rel = float(sim_to_query[i])
            sim_to_selected = max(float(pairwise[i, j]) for j in selected_idx) if selected_idx else 0.0
            mmr = lambda_param * rel - (1.0 - lambda_param) * sim_to_selected
            mmr_scores.append(mmr)
        next_idx = int(np.argmax(mmr_scores))
        selected_idx.append(next_idx)
        selected.append(candidate_ids[next_idx])
    return selected

# ---------------------------
# API endpoints
# ---------------------------
@router.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt required")
    idx = get_index()
    if idx is None:
        raise HTTPException(status_code=500, detail="index unavailable")

    # compute text embedding for query
    vecs = embed_texts([req.prompt], batch_size=32)
    if vecs.shape[0] == 0:
        raise HTTPException(status_code=500, detail="embedding failed")
    qv = vecs[0].astype(np.float32)

    top_k = int(req.top_k or 6)

    # If we have full local matrix (best path): do hybrid ranking
    if hasattr(idx, "matrix") and getattr(idx, "matrix") is not None and hasattr(idx, "ids"):
        try:
            matrix = idx.matrix  # numpy array (N,D)
            ids = idx.ids        # list of ids aligned with rows of matrix
            metadata = getattr(idx, "metadata", {}) or {}

            matrix_norm = l2_normalize_rows_np(matrix.astype(np.float32))
            qv_norm = qv / (np.linalg.norm(qv) + 1e-12)

            sims = cosine_similarities(qv_norm, matrix_norm)  # (N,)

            # meta-based boosts (category/brand)
            query_meta = getattr(req, "product_meta", None) or None
            if query_meta:
                sims = boost_by_meta(sims, ids, metadata, query_meta)

            # exact text match boosts (helps anchoring to keywords)
            exact_boosts = text_exact_match_boost(req.prompt, ids, metadata)
            sims = sims + exact_boosts

            # If there are many exact matches and they cover top_k, prefer them: quick check
            exact_count = int((exact_boosts > 0).sum())
            if exact_count >= top_k:
                # filter to exact matches only and rank by sims
                exact_idx = np.where(exact_boosts > 0)[0]
                order = exact_idx[np.argsort(-sims[exact_idx])]
                selected_pids = [ids[i] for i in order[:top_k]]
                out = [RecommendResponseItem(id=pid, score=float(sims[ids.index(pid)]), meta=metadata.get(pid, {})) for pid in selected_pids]
                return RecommendResponse(results=out)

            # Pre-select a larger pool for MMR
            M = min(matrix_norm.shape[0], max(M_PRESELECT, top_k * 8))
            if M >= sims.shape[0]:
                top_idx_all = np.argsort(-sims)
            else:
                top_idx = np.argpartition(-sims, M - 1)[:M]
                top_idx_all = top_idx[np.argsort(-sims[top_idx])]

            cand_ids = [ids[i] for i in top_idx_all]
            cand_vecs = matrix_norm[top_idx_all]

            # MMR to pick diverse high-similarity results
            selected_ids = mmr_rerank(qv_norm, cand_ids, cand_vecs, top_k=top_k, lambda_param=MMR_LAMBDA)

            # Build response
            pid_to_idx = {pid: i for i, pid in enumerate(ids)}
            out = []
            for pid in selected_ids:
                i_idx = pid_to_idx.get(pid)
                score = float(sims[i_idx]) if (i_idx is not None and i_idx < sims.shape[0]) else 0.0
                meta = metadata.get(pid, {})
                out.append(RecommendResponseItem(id=pid, score=score, meta=meta))
            return RecommendResponse(results=out)
        except Exception as e:
            # If anything goes wrong, fallback to idx.search
            # (fastapi will log exceptions)
            pass

    # Fallback: use idx.search (works for Pinecone or other wrappers)
    try:
        raw = idx.search(qv, top_k=top_k)
        out = []
        for item in raw:
            if len(item) == 3:
                pid, score, meta = item
            else:
                pid, score = item
                meta = getattr(idx, "metadata", {}).get(pid, {})
            out.append(RecommendResponseItem(id=pid, score=score, meta=meta))
        return RecommendResponse(results=out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search failed: {e}")


@router.get("/recommend/{product_id}", response_model=RecommendResponse)
async def recommend_by_product(product_id: str, top_k: int = 6):
    idx = get_index()
    if idx is None:
        raise HTTPException(status_code=500, detail="index unavailable")

    # Local matrix-based product-to-product recommend
    if hasattr(idx, "ids") and getattr(idx, "ids"):
        try:
            try:
                i = idx.ids.index(product_id)
            except ValueError:
                raise HTTPException(status_code=404, detail="product_id not found")
            matrix = idx.matrix
            metadata = getattr(idx, "metadata", {}) or {}

            matrix_norm = l2_normalize_rows_np(matrix.astype(np.float32))
            qv = matrix_norm[i]

            sims = cosine_similarities(qv, matrix_norm)

            # boost using product meta
            product_meta = metadata.get(product_id)
            if product_meta:
                sims = boost_by_meta(sims, idx.ids, metadata, product_meta)

            sims[i] = -999.0  # remove self

            # preselect candidates and MMR
            M = min(matrix_norm.shape[0], max(M_PRESELECT, top_k * 8))
            if M >= sims.shape[0]:
                top_idx_all = np.argsort(-sims)
            else:
                top_idx = np.argpartition(-sims, M - 1)[:M]
                top_idx_all = top_idx[np.argsort(-sims[top_idx])]

            cand_ids = [idx.ids[j] for j in top_idx_all]
            cand_vecs = matrix_norm[top_idx_all]

            selected_ids = mmr_rerank(qv, cand_ids, cand_vecs, top_k=top_k, lambda_param=MMR_LAMBDA)

            out = []
            for pid in selected_ids:
                meta = metadata.get(pid, {})
                score = float(sims[idx.ids.index(pid)]) if pid in idx.ids else 0.0
                out.append(RecommendResponseItem(id=pid, score=score, meta=meta))
            return RecommendResponse(results=out)
        except HTTPException:
            raise
        except Exception as e:
            pass

    # Pinecone fallback / other wrappers
    try:
        if hasattr(idx, "index") and getattr(idx, "initialized", False):
            resp = idx.index.fetch(ids=[product_id], include_metadata=True)
            vec_meta = resp.get("vectors", {}).get(product_id)
            if not vec_meta:
                raise HTTPException(status_code=404, detail="product_id not found in vector index")
            v = vec_meta.get("vector")
            if v is None:
                raise HTTPException(status_code=500, detail="product vector not available for product_id search")
            qv = np.array(v, dtype=np.float32)
            raw = idx.search(qv, top_k=top_k + 1)
            filtered = []
            for item in raw:
                if len(item) == 3:
                    pid, score, meta = item
                else:
                    pid, score = item
                    meta = {}
                if pid == product_id:
                    continue
                filtered.append((pid, score, meta))
                if len(filtered) >= top_k:
                    break
            out = [RecommendResponseItem(id=pid, score=score, meta=meta) for pid, score, meta in filtered]
            return RecommendResponse(results=out)
        else:
            raise HTTPException(status_code=400, detail="cannot perform product-based recommend on this index")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error: {e}")
