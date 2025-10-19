# scripts/nlp_cluster.py
"""
Compute product text embeddings (re-uses your chunking code) and cluster them with HDBSCAN.
Outputs app/data/nlp_groups.json with:
{
  "pid_to_cluster": {"pid1": 0, ...},
  "clusters": {
     "0": {"size": 123, "keywords": ["wooden","chair","dining"], "examples": ["pid1","pid2",...]},
     ...
  }
}
"""
import json
from pathlib import Path
import numpy as np
from collections import Counter
from tqdm import tqdm

# sklearn / sentence-transformers already in requirements
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer

# adjust paths
DATA_META = Path("app/data/index.npz.meta.json")
OUT = Path("app/data/nlp_groups.json")
CSV = Path("data/intern_data_ikarus.csv")  # if you want original csv

# load metadata and texts (we assume index metadata includes title + description)
meta = json.loads(DATA_META.read_text(encoding="utf-8"))
pids = list(meta.keys())

texts = []
for pid in pids:
    m = meta[pid]
    t = (m.get("title") or "") + ". " + (m.get("description") or "")
    texts.append(t)

print("Computing SBERT embeddings...")
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")  # small & fast
embs = model.encode(texts, show_progress_bar=True, batch_size=64, convert_to_numpy=True)

print("Clustering with HDBSCAN")
clusterer = hdbscan.HDBSCAN(min_cluster_size=8, metric='euclidean', cluster_selection_method='eom')
labels = clusterer.fit_predict(embs)  # -1 is noise

pid_to_cluster = {pid: int(lbl) for pid, lbl in zip(pids, labels)}

# compute keywords per cluster using TF-IDF
clusters = {}
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
X = vectorizer.fit_transform(texts)
terms = vectorizer.get_feature_names_out()

cluster_to_indices = {}
for idx, lbl in enumerate(labels):
    cluster_to_indices.setdefault(int(lbl), []).append(idx)

for lbl, indices in cluster_to_indices.items():
    if lbl == -1:
        continue
    # aggregate tfidf weights across cluster and take top terms
    sub = X[indices].sum(axis=0).A1
    topn = [terms[i] for i in np.argsort(-sub)[:8]]
    examples = [pids[i] for i in indices[:6]]
    clusters[str(lbl)] = {"size": len(indices), "keywords": topn, "examples": examples}

out = {"pid_to_cluster": pid_to_cluster, "clusters": clusters}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote NLP groups to", OUT)
