from fastapi import APIRouter, File, UploadFile, HTTPException
from app.modelsfolder.ml_models import load_cv_knn
from app.services.clip_embed import embed_image_bytes
import numpy as np

router = APIRouter(prefix="/cv")


@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    vec = embed_image_bytes(data)
    if vec is None:
        raise HTTPException(status_code=400, detail="Embedding failed.")
    vec = vec.reshape(1, -1)
    knn_obj = load_cv_knn()
    knn = knn_obj["knn"]
    le = knn_obj["label_encoder"]
    probs = knn.predict_proba(vec)[0]
    top_idx = np.argsort(-probs)[:5]
    top_labels = [(le.classes_[i], float(probs[i])) for i in top_idx]
    return {"predictions": top_labels}
