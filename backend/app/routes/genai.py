# app/routes/genai.py
from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

LOG = logging.getLogger("uvicorn.error")

# Try to import the generator wrapper.
try:
    from app.modelsfolder.ml_models import generate_text, load_genai
    _HAS_GEN = True
except Exception as e:
    LOG.warning("Could not import app.modelsfolder.ml_models.generate_text: %s", e)
    _HAS_GEN = False

router = APIRouter(prefix="/gen", tags=["genai"])

class GenByProduct(BaseModel):
    product_id: Optional[str] = None
    title: Optional[str] = ""
    description: Optional[str] = ""
    meta: Optional[Dict[str, Any]] = None
    max_length: Optional[int] = 120
    style: Optional[str] = "creative"
    temperature: Optional[float] = None
    top_p: Optional[float] = None

@router.post("/description")
def gen_description(payload: GenByProduct, request: Request, debug: bool = Query(False)):
    """
    Generate a product description.
    Sends *title* to generate_text so that generate_text composes a few-shot prompt.
    Optional query param `?debug=true` will return the final prompt as well (useful for debugging).
    """
    try:
        title = (payload.title or "").strip()
        desc = (payload.description or "").strip()
        meta = payload.meta or {}
        max_length = int(payload.max_length or 120)
        style = payload.style or "creative"
        temperature = payload.temperature
        top_p = payload.top_p
        cache_key = None
        if payload.product_id:
            cache_key = f"gen:{payload.product_id}"

        # Compose a short input for generate_text: pass the title (not a full prompt)
        # so the generator can add few-shot examples and meta internally.
        # We'll pass the description/meta via style param by encoding in the prompt text when needed
        # but better: let generate_text take a simple title and style, it will build full prompt.
        # If you want to include description/meta, we include them appended to the title separated by " | "
        if desc:
            short_input = f"{title} | {desc}"
        else:
            short_input = title

        LOG.debug("Gen request payload (product_id=%s) title=%s max_length=%d style=%s",
                  payload.product_id, title[:120], max_length, style)

        if _HAS_GEN:
            # ensure model is loaded (idempotent)
            try:
                load_genai()
            except Exception as e:
                LOG.debug("load_genai() raised: %s", e)

            # call generate_text: let it create few-shot prompt and use sampling params
            gen_kwargs = {"max_length": max_length, "style": style, "cache_key": cache_key}
            if temperature is not None:
                gen_kwargs["temperature"] = float(temperature)
            if top_p is not None:
                gen_kwargs["top_p"] = float(top_p)

            LOG.debug("Calling generate_text with args: %s", gen_kwargs)
            out = generate_text(short_input, **gen_kwargs)
            if not isinstance(out, str):
                out = str(out)

        else:
            from . import genai as fallback  # local fallback
            LOG.debug("Using fallback generator")
            out = fallback._fallback_generate_text(short_input, max_length=max_length)

        out_clean = out.strip()
        LOG.info("Generated text length=%d for product_id=%s", len(out_clean), payload.product_id)

        # If debug requested, return also the internal prompt used (if generate_text exposes it)
        resp = {"description": out_clean}
        if debug:
            resp["_debug_note"] = "Sent short input to generate_text; generate_text composes full prompt internally."
            resp["_short_input"] = short_input
        return resp

    except HTTPException:
        raise
    except Exception as e:
        LOG.exception("Generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
