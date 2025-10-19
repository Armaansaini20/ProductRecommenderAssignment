# scripts/upsert_to_pinecone.py
"""
Run this once to push your local index (index.npz + index.npz.meta.json)
to a Pinecone index. Set env vars PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME.
"""

import os
import json
from pathlib import Path
import numpy as np
import pinecone

# Config (defaults)
INDEX_PATH = Path(os.getenv("INDEX_PATH", "app/data/index.npz"))
META_PATH = Path(os.getenv("INDEX_META_PATH", "app/data/index.npz.meta.json"))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ikarus-products")
BATCH_SIZE = int(os.getenv("PINECONE_UPSERT_BATCH", "100"))

if not PINECONE_API_KEY or not PINECONE_ENV:
    raise RuntimeError("Set PINECONE_API_KEY and PINECONE_ENV environment variables before running this script.")

print("Loading local index:", INDEX_PATH, META_PATH)
data = np.load(str(INDEX_PATH), allow_pickle=True)
ids = [str(x) for x in data["ids"].tolist()]
matrix = data["matrix"].astype(np.float32)
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Initing Pinecone:", PINECONE_ENV)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# create index if doesn't exist
if INDEX_NAME not in pinecone.list_indexes():
    print("Creating Pinecone index:", INDEX_NAME)
    pinecone.create_index(name=INDEX_NAME, dimension=matrix.shape[1], metric="cosine")
else:
    print("Pinecone index exists:", INDEX_NAME)

idx = pinecone.Index(INDEX_NAME)

n = len(ids)
print(f"Upserting {n} vectors in batches of {BATCH_SIZE}...")

for i in range(0, n, BATCH_SIZE):
    j = min(i + BATCH_SIZE, n)
    batch = []
    for k in range(i, j):
        pid = ids[k]
        vec = matrix[k].astype(float).tolist()
        meta = metadata.get(pid, {})
        # keep metadata small and serializable
        batch.append((pid, vec, meta))
    # upsert batch (will overwrite if id exists)
    idx.upsert(vectors=batch)
    print(f"Upserted {i}-{j}")

print("Upsert completed.")
