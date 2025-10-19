"""
train_cv_knn.py
---------------
Train a KNN classifier on product image embeddings to predict category.
Uses cosine distance between normalized CLIP embeddings.
Saves app/data/cv_knn.pkl
"""

import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score
from collections import Counter
import ast
import joblib

INDEX = Path("app/data/index.npz")
META = Path("app/data/index.npz.meta.json")
OUT = Path("app/data/cv_knn.pkl")

# === Load data ===
print("Loading embeddings...")
data = np.load(str(INDEX), allow_pickle=True)
ids = [str(x) for x in data["ids"].tolist()]
matrix = data["matrix"].astype(np.float32)
meta = json.loads(META.read_text(encoding="utf-8"))

# === Category extraction helper ===
def extract_label_from_meta(m, coarsen_levels=1):
    raw = m.get("categories") or m.get("category") or m.get("categories_field") or None
    if raw is None:
        return None
    if isinstance(raw, list):
        parts = [str(x).strip() for x in raw if x]
    else:
        s = str(raw).strip()
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                parts = [str(x).strip() for x in parsed if x]
            else:
                parts = [s]
        except Exception:
            if ">" in s:
                parts = [p.strip() for p in s.split(">")]
            else:
                parts = [p.strip() for p in s.split(",")]
    if not parts:
        return None
    selected = parts[-coarsen_levels:]
    return " > ".join(selected)

# === Build dataset ===
labels = []
X = []
for i, pid in enumerate(ids):
    m = meta.get(pid, {})
    lab = extract_label_from_meta(m, coarsen_levels=2)
    if not lab:
        continue
    labels.append(lab)
    X.append(matrix[i])
X = np.array(X)
y = np.array(labels)
print(f"Loaded {len(X)} samples.")

# === Clean rare categories ===
MIN_SAMPLES = 3
cnt = Counter(y)
rare = {c for c, v in cnt.items() if v < MIN_SAMPLES}
if rare:
    print(f"Merging {len(rare)} rare categories into OTHER.")
    y = np.array([lab if lab not in rare else "OTHER" for lab in y])

le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"Classes: {len(le.classes_)}")

# === Train/test split ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.15, random_state=42
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# === Train KNN ===
knn = KNeighborsClassifier(
    n_neighbors=5, metric="cosine", weights="distance", n_jobs=-1
)
knn.fit(X_train, y_train)

# === Evaluate ===
def evaluate_topk(model, X, y, ks=[1, 3, 5]):
    probs = model.predict_proba(X)
    for k in ks:
        acc = top_k_accuracy_score(y, probs, k=k, labels=np.arange(len(le.classes_)))
        print(f"Top-{k} accuracy: {acc:.4f}")

print("\nValidation performance:")
evaluate_topk(knn, X_val, y_val)

# === Save model ===
joblib.dump({"knn": knn, "label_encoder": le}, OUT)
print(f"✅ Saved KNN model to {OUT}")
