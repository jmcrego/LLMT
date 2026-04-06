import ollama

from fastapi import HTTPException
from pydantic import BaseModel
from .shared import model_store, model_lock


class LLMTUploadRequest(BaseModel):
    model: str


class LLMTUploadResponse(BaseModel):
    status: str
    model: str


def upload_endpoint(request: LLMTUploadRequest) -> LLMTUploadResponse:
    model = request.model
    try:
        ollama.pull(model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model '{model}': {e}")
    with model_lock:
        model_store["model"] = model
    return LLMTUploadResponse(status="ok", model=model)
