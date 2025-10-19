# app/routes/search.py
from fastapi import APIRouter, File, UploadFile, HTTPException
from app.services.clip_embed import embed_image_bytes, load_clip
from app.services.indexer import get_index
from app.models import RecommendResponse, RecommendResponseItem

router = APIRouter()

# ensure CLIP is available lazily
try:
    load_clip()
except Exception:
    pass

@router.post("/search/image", response_model=RecommendResponse)
async def search_image(file: UploadFile = File(...), top_k: int = 6):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    try:
        vec = embed_image_bytes(data)   # normalized numpy vector
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image: {e}")
    idx = get_index()
    if idx is None:
        raise HTTPException(status_code=500, detail="index unavailable")
    raw = idx.search(vec, top_k=top_k)
    out = []
    for item in raw:
        if len(item) == 3:
            pid, score, meta = item
        else:
            pid, score = item
            meta = getattr(idx, "metadata", {}).get(pid, {})
        # try to expose first image URL key in meta for frontend
        image_url = meta.get("image_url") or meta.get("images") or meta.get("image") or meta.get("images_field")
        if isinstance(image_url, list):
            image_url = image_url[0] if image_url else None
        meta["image_url"] = image_url
        out.append(RecommendResponseItem(id=pid, score=score, meta=meta))
    return RecommendResponse(results=out)
