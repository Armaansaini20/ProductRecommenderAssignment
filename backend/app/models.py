# app/models.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class RecommendRequest(BaseModel):
    prompt: Optional[str] = None
    top_k: Optional[int] = 6

class RecommendResponseItem(BaseModel):
    id: str
    score: float
    meta: Optional[Dict[str, Any]] = None

class RecommendResponse(BaseModel):
    results: List[RecommendResponseItem]
