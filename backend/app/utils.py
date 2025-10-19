import json
from pathlib import Path
from .services.indexer import get_index

DATA_PATH = Path(__file__).parent / "data" / "products.json"

def load_products():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_initial_index(emb_func):
    """
    emb_func: Callable[[List[str]], List[List[float]]]
    loads products.json, computes embeddings (on title+description), adds to index and builds it.
    """
    products = load_products()
    texts = [p["title"] + ". " + p.get("description","") for p in products]
    vectors = emb_func(texts)
    idx = get_index()
    for p, v in zip(products, vectors):
        idx.add(p["id"], v, p)
    idx.build()
    return products
