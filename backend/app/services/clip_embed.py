# app/services/clip_embed.py
from typing import List
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

_MODEL = None
_PROCESSOR = None
_DEVICE = None

def load_clip(model_name: str = "openai/clip-vit-base-patch32", device: str = None):
    global _MODEL, _PROCESSOR, _DEVICE
    if _MODEL is not None:
        return
    _DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _PROCESSOR = CLIPProcessor.from_pretrained(model_name)
    _MODEL = CLIPModel.from_pretrained(model_name).to(_DEVICE)
    _MODEL.eval()

def _l2norm_np(a: np.ndarray):
    if a.ndim == 2:
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        return a / norms
    else:
        n = np.linalg.norm(a)
        return a / (n if n!=0 else 1.0)

def embed_texts(texts: List[str], device: str = None, batch_size: int = 32) -> np.ndarray:
    load_clip(device=device)
    device = _DEVICE
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = _PROCESSOR(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
            feats = _MODEL.get_text_features(**inputs)
            out.append(feats.cpu().numpy())
    arr = np.vstack(out).astype(np.float32) if out else np.zeros((0, _MODEL.config.projection_dim), dtype=np.float32)
    return _l2norm_np(arr)

def embed_image_bytes(img_bytes: bytes, device: str = None) -> np.ndarray:
    load_clip(device=device)
    device = _DEVICE
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    inputs = _PROCESSOR(images=img, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        feats = _MODEL.get_image_features(**inputs)
    arr = feats.cpu().numpy().reshape(-1).astype(np.float32)
    arr = arr / (np.linalg.norm(arr) + 1e-12)
    return arr
