# app/routes/analytics.py
from fastapi import APIRouter
import json
from pathlib import Path
from collections import Counter
from typing import Dict, Any

router = APIRouter(prefix="/analytics")
ANALYTICS = Path("app/data/analytics.json")
META_PATH = Path("app/data/index.npz.meta.json")

def _safe_get_first_category(cat_val):
    # Accept list or string, return a normalized string
    if cat_val is None:
        return "unknown"
    if isinstance(cat_val, (list, tuple)):
        if not cat_val:
            return "unknown"
        # If nested lists/strings, convert to string of first element
        first = cat_val[0]
        # If first is list itself (rare), join
        if isinstance(first, (list, tuple)):
            return " / ".join(map(str, first))
        return str(first)
    # sometimes it's a string that looks like list; just return it
    return str(cat_val)

@router.get("/summary")
def summary() -> Dict[str, Any]:
    """
    Return a stable analytics summary schema that the frontend expects.

    Schema (example):
    {
      "total_products": 312,
      "products_with_images": 200,
      "unique_categories": 52,
      "top_categories": [["chairs", 68], ["sofas", 34], ...],
      "by_category": {"chairs": 68, "sofas":34, ...},
      "price_count": 120,
      "raw_meta_sample": { "<pid>": { ... } }  <-- optional small sample for debugging
    }
    """

    # If precomputed analytics file exists, try to read and adapt it
    if ANALYTICS.exists():
        try:
            data = json.loads(ANALYTICS.read_text(encoding="utf-8"))
            # Try to normalize / adapt if needed
            # If already matches expected field names, return directly
            if any(k in data for k in ("total_products", "products_with_images", "top_categories")):
                return data
            # Otherwise attempt to map old keys to new schema
            mapped = {}
            mapped["total_products"] = data.get("n_products") or data.get("total_products") or len(data.get("by_category", {}))
            mapped["products_with_images"] = data.get("images") or data.get("products_with_images") or 0
            by_cat = data.get("by_category") or data.get("by_category", {}) or data.get("by_category", {})
            # ensure dict
            if isinstance(by_cat, list):
                # maybe list of pairs
                by_cat = dict(by_cat)
            mapped["by_category"] = by_cat
            # produce top_categories list
            top = sorted(by_cat.items(), key=lambda x: -x[1]) if by_cat else []
            mapped["top_categories"] = top
            mapped["unique_categories"] = len(by_cat)
            mapped["price_count"] = data.get("price_count") or data.get("price_count", 0)
            mapped["raw"] = data
            return mapped
        except Exception:
            # fall through to best-effort compute
            pass

    # Otherwise compute from meta file
    if not META_PATH.exists():
        # If nothing exists, return an empty but valid structure
        return {
            "total_products": 0,
            "products_with_images": 0,
            "unique_categories": 0,
            "top_categories": [],
            "by_category": {},
            "price_count": 0,
            "raw_meta_sample": {}
        }

    meta_all = json.loads(META_PATH.read_text(encoding="utf-8"))
    cats = Counter()
    images = 0
    prices = 0
    price_count = 0

    # Build a small sample for raw_meta_sample to help debugging in frontend
    sample = {}
    sample_limit = 6

    for pid, m in meta_all.items():
        # detect categories from various possible keys
        cat_val = m.get("categories") or m.get("category") or m.get("categories_field") or m.get("category_field") or None
        cat_norm = _safe_get_first_category(cat_val)
        cats[cat_norm] += 1

        # detect image presence from common fields used in your code
        if m.get("local_image") or m.get("image_url") or m.get("images") or m.get("image"):
            images += 1

        p = m.get("price")
        try:
            if p is not None and p != "":
                _ = float(p)
                price_count += 1
        except Exception:
            pass

        if len(sample) < sample_limit:
            sample[pid] = {
                "title": m.get("title"),
                "image": m.get("image_url") or m.get("local_image") or (m.get("images")[0] if isinstance(m.get("images"), (list,tuple)) and m.get("images") else None),
                "categories": m.get("categories")
            }

    by_category = dict(cats)
    top_categories = sorted(by_category.items(), key=lambda kv: -kv[1])

    result = {
        "total_products": len(meta_all),
        "products_with_images": images,
        "unique_categories": len(by_category),
        "top_categories": top_categories,           # array of [name, count]
        "by_category": by_category,                 # mapping for completeness
        "price_count": price_count,
        "raw_meta_sample": sample
    }

    return result
