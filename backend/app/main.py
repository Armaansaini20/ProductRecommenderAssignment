# app/main.py
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import nlp as nlp_router, cv as cv_router, genai as genai_router, analytics as analytics_router

# Import your routers (adjust paths if your project uses different module names)
from app.routes import recommend as recommend_router
from app.routes import search as search_router

# indexer factory and clip helper
from app.services.indexer import get_index
from app.services.clip_embed import load_clip

# basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ikarus3d")

# read env
USE_PINECONE = os.getenv("USE_PINECONE", "false").lower() in ("1", "true", "yes")
INDEX_PATH = os.getenv("INDEX_PATH", "app/data/index.npz")
INDEX_META = os.getenv("INDEX_META_PATH", "app/data/index.npz.meta.json")
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
# allowed origins for CORS during development; update for production
_allowed = os.getenv("FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000")
ALLOW_ORIGINS = [s.strip() for s in _allowed.split(",") if s.strip()]

app = FastAPI(title="Ikarus3D - Product Recommender", version="1.0.0")

# CORS middleware (adjust allow_origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include routers
app.include_router(recommend_router.router)
app.include_router(search_router.router)
app.include_router(nlp_router.router)
app.include_router(cv_router.router)
app.include_router(genai_router.router)
app.include_router(analytics_router.router)


@app.on_event("startup")
def startup_event():
    """
    Initialize index (either local numpy or Pinecone) and ensure CLIP is ready (lazy load allowed).
    We avoid crashing the server on errors here so the process can start; endpoints will return errors if
    index/CLIP truly unavailable.
    """
    logger.info("Starting Ikarus3D backend (USE_PINECONE=%s)", USE_PINECONE)
    idx = get_index()
    try:
        if USE_PINECONE:
            # PineconeIndex exposes .init()
            try:
                idx.init()
                logger.info("Pinecone index initialized (name=%s)", getattr(idx, "name", "unknown"))
            except Exception as ex:
                logger.error("Pinecone index initialization failed: %s", ex)
        else:
            # VectorIndex exposes .load(npz_path, meta_path)
            try:
                info = idx.load(INDEX_PATH, INDEX_META)
                logger.info("Local index loaded: items=%s dim=%s", info.get("n_items"), info.get("dim"))
            except FileNotFoundError as fnf:
                logger.error("Local index file not found: %s", fnf)
            except Exception as ex:
                logger.error("Failed to load local index: %s", ex)
    except Exception as e:
        logger.exception("Unexpected error during index setup: %s", e)

    # Try to warm CLIP (this is optional - the library functions will lazy-load if needed)
    try:
        load_clip(model_name=CLIP_MODEL_NAME)
        logger.info("CLIP model available (requested=%s).", CLIP_MODEL_NAME)
    except Exception as e:
        # don't crash the server if CLIP can't be loaded now; endpoints will error when used
        logger.warning("CLIP model load warning: %s", e)


@app.get("/health")
def health():
    """
    Simple health endpoint for load balancers / monitoring.
    Returns whether an index is present and a rough item count if available.
    """
    idx = get_index()
    result = {"ok": True, "use_pinecone": USE_PINECONE}
    try:
        if not USE_PINECONE and hasattr(idx, "ids"):
            result["index_items"] = len(get_index().ids)
        elif USE_PINECONE and hasattr(idx, "name"):
            result["index_name"] = getattr(idx, "name")
    except Exception:
        # best-effort; don't fail health check due to introspection errors
        logger.debug("Health endpoint index introspection failed", exc_info=True)
    return result
