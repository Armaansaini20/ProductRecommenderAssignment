# app/routes/nlp.py
from fastapi import APIRouter
import json
from pathlib import Path

router = APIRouter(prefix="/nlp")

DATA = Path("app/data/nlp_groups.json")
META = Path("app/data/index.npz.meta.json")

@router.get("/groups")
def get_groups():
    groups = json.loads(DATA.read_text(encoding="utf-8"))
    # optionally include short metadata samples
    meta = json.loads(META.read_text(encoding="utf-8"))
    for cid, info in groups.get("clusters", {}).items():
        ex = info.get("examples", [])[:6]
        info["examples_meta"] = [meta.get(pid, {}) for pid in ex]
    return groups
