# app/models/ml_models.py
from pathlib import Path
import torch
import torch.nn as nn
import json
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.neighbors import KNeighborsClassifier
import joblib

DATA_DIR = Path("app/data")
_knn_cv = None
def load_cv_knn(path=DATA_DIR / "cv_knn.pkl"):
    global _knn_cv
    if _knn_cv is not None:
        return _knn_cv
    d = joblib.load(path)
    _knn_cv = d
    return _knn_cv

# CV MLP loader
_cv = None
def load_cv_mlp(path=DATA_DIR / "cv_classifier.pt"):
    global _cv
    if _cv is not None:
        return _cv
    d = torch.load(str(path), map_location="cpu")
    dim = d["dim"]
    n_classes = len(d["label_encoder"].classes_)
    # recreate same model shape
    model = nn.Sequential(
        nn.Linear(dim, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, n_classes)
    )
    model.load_state_dict(d["model_state"])
    model.eval()
    _cv = {"model": model, "label_encoder": d["label_encoder"]}
    return _cv

# GenAI loader (Flan-T5 small)
_gen = None
def load_genai(model_name="google/flan-t5-small"):
    global _gen
    if _gen is not None:
        return _gen
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    _gen = {"tokenizer": tokenizer, "model": model}
    return _gen

def generate_text(prompt, max_length=64, model_name="google/flan-t5-small"):
    gen = load_genai(model_name)
    tok = gen["tokenizer"]
    m = gen["model"]
    inputs = tok(prompt, return_tensors="pt").to(next(m.parameters()).device)
    out = m.generate(**inputs, max_new_tokens=max_length, do_sample=True, top_p=0.95, temperature=0.9)
    return tok.decode(out[0], skip_special_tokens=True).strip()

# app/modelsfolder/ml_models.py
import threading
import logging
from typing import Optional
from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

LOG = logging.getLogger("uvicorn.error")

# Choose model here. If you have CPU-only and limited RAM, use a small model.
# If you have GPU / larger RAM, consider "google/flan-t5-large" or "google/flan-t5-xl" or an instruction-tuned LLM.
DEFAULT_MODEL_NAME = "google/flan-t5-small"  # change to flan-t5-base/large if available

# Module-level model/tokenizer cache + lock for thread-safety
_MODEL = None
_TOKENIZER = None
_MODEL_LOCK = threading.Lock()
# simple in-memory cache for generated descriptions keyed by (key)
_GENERATED_CACHE = {}

def load_genai(model_name: Optional[str] = None, device: Optional[str] = None):
    """
    Lazy-load model and tokenizer. Safe to call multiple times.
    Returns (tokenizer, model, device).
    """
    global _MODEL, _TOKENIZER
    with _MODEL_LOCK:
        if _MODEL is not None and _TOKENIZER is not None:
            # already loaded
            return _TOKENIZER, _MODEL, _device()

        model_name = model_name or DEFAULT_MODEL_NAME

        # Determine device: prefer GPU if available
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        LOG.info("Loading generation model '%s' on device=%s", model_name, device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # seq2seq model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.to(device)
        model.eval()

        _TOKENIZER = tokenizer
        _MODEL = model

        return tokenizer, model, device

def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def _compose_prompt(title: str, description: str, meta: dict = None, style: str = "creative"):
    """
    Compose a stronger instruction prompt with a few-shot examples and explicit style directions.
    """
    meta = meta or {}
    brand = meta.get("brand") or meta.get("manufacturer") or ""
    material = meta.get("material") or ""
    color = meta.get("color") or ""
    categories = meta.get("categories")
    categories_str = ""
    if categories:
        if isinstance(categories, (list, tuple)):
            categories_str = ", ".join(map(str, categories[:4]))
        else:
            categories_str = str(categories)

    # Few-shot examples — short and descriptive; this nudges the model to produce richer creative text
    examples = (
        "Example 1:\n"
        "Product: 'Nordic Oak Coffee Table'\n"
        "Meta: brand 'HausLine', material 'oak', color 'natural'\n"
        "Creative description: A warm Scandinavian-inspired coffee table in solid oak — clean lines and a soft matte finish create a calm focal point for casual mornings and cozy evenings. Its low profile and visible wood grain bring understated craftsmanship to any modern living room.\n\n"
        "Example 2:\n"
        "Product: 'Convertible Sofa Bed'\n"
        "Meta: brand 'UrbanSleep', material 'linen blend', color 'slate'\n"
        "Creative description: A versatile sofa that converts into a comfortable bed in seconds — tailored linen upholstery and slim tapered legs merge contemporary style with practical design. Ideal for chic studio living, it balances compact form with restful comfort.\n\n"
    )

    # Instruction
    inst = f"Write a short creative product design description (2-4 sentences) in an evocative, visual tone. Avoid boilerplate — show how it feels, looks, and where it fits.\n\nProduct: \"{title}\"\n"
    if description:
        inst += f"Short product info: {description.strip()}\n"
    if brand or material or color or categories_str:
        meta_bits = []
        if brand: meta_bits.append(f"brand: {brand}")
        if material: meta_bits.append(f"material: {material}")
        if color: meta_bits.append(f"color: {color}")
        if categories_str: meta_bits.append(f"categories: {categories_str}")
        inst += "Meta: " + "; ".join(meta_bits) + "\n"

    inst += f"Style: {style}. Keep it creative, sensory, and concise.\n\nReturn only the description (no headings)."

    # Put examples before instruction to serve as few-shot
    prompt = examples + "\n" + inst
    return prompt

def generate_text(
    instruction_or_prompt: str,
    max_length: int = 120,
    temperature: float = 0.9,
    top_p: float = 0.92,
    num_beams: int = 1,
    do_sample: bool = True,
    style: str = "creative",
    cache_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> str:
    """
    Generate a creative description string.

    - instruction_or_prompt: can be either a raw prompt or a product title (we'll detect).
    - max_length: approximate output length in tokens / new tokens (for seq2seq, we use max_new_tokens)
    - temperature/top_p/do_sample: sampling params for creativity
    - num_beams: if >1, beam search is used (typically for deterministic quality)
    - style: "creative", "technical", "short", etc. (affects prompt template)
    - cache_key: optional unique key to cache the result
    """

    # Simple cache lookup
    if cache_key and cache_key in _GENERATED_CACHE:
        return _GENERATED_CACHE[cache_key]

    tokenizer, model, device = load_genai(model_name=model_name)

    # Decide whether the input is a product title/desc or a full crafted prompt:
    # Heuristic: if the instruction contains "Product:" or "Meta:" or "Example", assume it's full prompt already.
    if "Product:" in instruction_or_prompt or "Meta:" in instruction_or_prompt or "Example" in instruction_or_prompt:
        prompt = instruction_or_prompt
    else:
        # If the string looks like a short title, create a richer prompt around it
        prompt = _compose_prompt(instruction_or_prompt, "", {}, style=style)

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Use generation params for more creativity
    gen_kwargs = dict(
        max_new_tokens=int(max_length),
        temperature=float(temperature),
        top_p=float(top_p),
        do_sample=bool(do_sample),
        num_return_sequences=1,
    )

    # If user requested beams and not sampling, set do_sample False and num_beams accordingly
    if num_beams and int(num_beams) > 1:
        gen_kwargs["num_beams"] = int(num_beams)
        gen_kwargs["do_sample"] = False

    try:
        with torch.no_grad():
            out_ids = model.generate(**inputs, **gen_kwargs)
            out_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    except Exception as e:
        LOG.exception("Generation failed: %s", e)
        # graceful fallback to simple template if generation fails
        out_text = _fallback_text(instruction_or_prompt)

    out_text = out_text.strip()

    # Cache the result if key provided
    if cache_key:
        _GENERATED_CACHE[cache_key] = out_text

    return out_text

def _fallback_text(prompt:str):
    # Very short fallback
    return "A thoughtfully designed product that blends form and function — ideal for everyday living."

# small helper to clear cache (useful while debugging)
def clear_generation_cache():
    _GENERATED_CACHE.clear()
