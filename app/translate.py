import time
import ollama

from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Optional
from .shared import model_store, model_lock


class TermPair(BaseModel):
    source: str
    target: str


class SimilarTranslation(BaseModel):
    source: str
    target: str


class LLMTTranslateRequest(BaseModel):
    sentence: str
    target_language: str
    previous_sentence: Optional[str] = None
    terminology: Optional[List[TermPair]] = None
    similar_translations: Optional[List[SimilarTranslation]] = None


class LLMTTranslateResponse(BaseModel):
    translation: str
    model: str
    runtime_ms: float = 0


def build_prompt(request: LLMTTranslateRequest) -> str:
    parts = []
    if request.previous_sentence:
        parts.append(f"Previous sentence: {request.previous_sentence}")
    if request.terminology:
        terms = "; ".join(f"{t.source} → {t.target}" for t in request.terminology)
        parts.append(f"Terminology: {terms}")
    if request.similar_translations:
        examples = "\n".join(f"  {st.source} → {st.target}" for st in request.similar_translations)
        parts.append(f"Similar translations:\n{examples}")
    parts.append(
        f"Translate the following sentence into {request.target_language}. "
        f"Output only the translation, with no extra text:\n{request.sentence}"
    )
    return "\n\n".join(parts)


def translate_endpoint(request: LLMTTranslateRequest) -> LLMTTranslateResponse:
    tic = time.perf_counter()
    with model_lock:
        model = model_store["model"]
    prompt = build_prompt(request)
    try:
        response = ollama.generate(model=model, prompt=prompt)
        translation = response.get("response") or ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}")
    runtime_s = time.perf_counter() - tic
    return LLMTTranslateResponse(
        translation=translation,
        model=model,
        runtime_ms=runtime_s * 1000,
    )
