# scripts/clean_image_urls_in_meta.py
"""
Cleans image fields in app/data/index.npz.meta.json by extracting a single
clean image_url per product.

Usage:
  (activate venv)
  python scripts/clean_image_urls_in_meta.py
"""

import ast
import json
import re
from pathlib import Path

META_PATH = Path("app/data/index.npz.meta.json")  # adjust if different

if not META_PATH.exists():
    raise SystemExit(f"Meta file not found: {META_PATH}")

def extract_first_url(val):
    """Return a cleaned URL string or None."""
    if not val:
        return None
    # If already a list/tuple
    if isinstance(val, (list, tuple)):
        for v in val:
            if isinstance(v, str) and v.strip().lower().startswith("http"):
                return v.strip()
        return None
    # If it's a dict with common fields
    if isinstance(val, dict):
        for key in ("image_url", "images", "images_field", "image"):
            v = val.get(key)
            if v:
                r = extract_first_url(v)
                if r:
                    return r
        return None
    # If it's a plain string
    s = str(val).strip()
    # Try literal eval for stringified list: "['http...', 'http...']"
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return extract_first_url(parsed)
        except Exception:
            pass
    # Regex: find first http(s) URL in the string
    m = re.search(r"(https?://[^\s'\",\\\]]+)", s)
    if m:
        url = m.group(1).strip().strip("',\"")
        return url
    return None

print("Loading metadata:", META_PATH)
data = json.loads(META_PATH.read_text(encoding="utf-8"))

updated = 0
total = len(data)
for pid, meta in list(data.items()):
    # try common fields that might contain images
    candidates = []
    for f in ("image_url", "images", "images_field", "image", "local_image"):
        if f in meta:
            candidates.append(meta[f])
    # also check meta itself if it was saved as a stringified representation
    if not candidates and isinstance(meta, str):
        candidates = [meta]

    first = None
    for cand in candidates:
        first = extract_first_url(cand)
        if first:
            break

    # fallback: check nested keys or "images_field" within a stringified dict
    if not first:
        # try to search in string representation of meta
        first = extract_first_url(str(meta))

    if first:
        # normalize: remove surrounding quotes and spaces
        norm = first.strip().strip('"\'')

        # store only if different or missing
        if meta.get("image_url") != norm:
            meta["image_url"] = norm
            data[pid] = meta
            updated += 1

print(f"Total items: {total}, Updated image_url for: {updated}")
# backup old file
bak = META_PATH.with_suffix(META_PATH.suffix + ".bak")
META_PATH.rename(bak)
print("Backup saved to", bak)
META_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote cleaned metadata to", META_PATH)
