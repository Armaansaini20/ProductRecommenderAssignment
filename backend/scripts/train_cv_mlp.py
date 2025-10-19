# scripts/train_cv_mlp.py
"""
Train a small MLP on image embeddings to predict category.
Requires: an "index.npz" with ids and matrix, and index meta json with "categories" field.
Saves app/data/cv_classifier.pt
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

INDEX = Path("app/data/index.npz")
META = Path("app/data/index.npz.meta.json")
OUT = Path("app/data/cv_classifier.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = np.load(str(INDEX), allow_pickle=True)
ids = [str(x) for x in data["ids"].tolist()]
matrix = data["matrix"].astype(np.float32)
meta = json.loads(META.read_text(encoding="utf-8"))

labels = []
X = []
for i, pid in enumerate(ids):
    m = meta.get(pid, {})
    cat = m.get("categories") or m.get("category") or m.get("categories_field") or None
    if not cat:
        labels.append(None)
        X.append(matrix[i])
    else:
        # if categories are lists or stringified, get first token
        if isinstance(cat, list):
            lab = str(cat[0])
        else:
            lab = str(cat)
        labels.append(lab)
        X.append(matrix[i])

# filter items with labels
X = np.array(X)
y = np.array(labels)
mask = [i for i, v in enumerate(y) if v is not None and v != ""]
X = X[mask]
y = y[mask]

le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Classes:", le.classes_, "n_classes=", len(le.classes_))

from collections import Counter

# Filter out classes with < 2 samples
counts = Counter(y_enc)
valid_idx = [i for i, yv in enumerate(y_enc) if counts[yv] >= 2]
invalid = len(y_enc) - len(valid_idx)

if invalid > 0:
    print(f"⚠️ Skipping {invalid} samples from categories with <2 items.")
    X = X[valid_idx]
    y_enc = y_enc[valid_idx]

# If still enough variety, stratify; else fallback to random split
if len(set(y_enc)) > 1:
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_enc, test_size=0.15, random_state=42, stratify=y_enc
        )
    except ValueError as e:
        print("⚠️ Stratified split failed, falling back to random split:", e)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_enc, test_size=0.15, random_state=42
        )
else:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc, test_size=0.15, random_state=42
    )

print("Train/val sizes:", X_train.shape, X_val.shape)


class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_ds = EmbeddingDataset(X_train, y_train)
val_ds = EmbeddingDataset(X_val, y_val)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128)

# tiny MLP
n_dim = X_train.shape[1]
n_classes = len(le.classes_)
model = nn.Sequential(
    nn.Linear(n_dim, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, n_classes)
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

def train_epoch():
    model.train()
    total=0; acc=0; loss_sum=0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += float(loss.item())
        total += yb.size(0)
        acc += (logits.argmax(1)==yb).sum().item()
    return loss_sum/len(train_loader), acc/total

def eval_epoch():
    model.eval()
    total=0; acc=0; loss_sum=0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss_sum += float(loss.item())
            total += yb.size(0)
            acc += (logits.argmax(1)==yb).sum().item()
    return loss_sum/len(val_loader), acc/total

best_val=0
for epoch in range(1, 21):
    tr_loss, tr_acc = train_epoch()
    val_loss, val_acc = eval_epoch()
    print(f"Epoch {epoch}: tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} val_acc={val_acc:.4f}")
    if val_acc > best_val:
        best_val = val_acc
        torch.save({"model_state": model.state_dict(), "label_encoder": le, "dim": n_dim}, str(OUT))
        print("Saved best model with val_acc", best_val)

print("Training complete, best_val:", best_val)
