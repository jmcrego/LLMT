import time
import ollama
import logging

from fastapi import HTTPException
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
from .shared import model_store, model_lock

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(Path("translate.log") , mode='a', encoding='utf-8'),
        # logging.StreamHandler()  # and log to console
    ]
)

logger = logging.getLogger("translate")

class TermPair(BaseModel):
    source: str
    target: str


class SimilarTranslation(BaseModel):
    source: str
    target: str

class Context(BaseModel):
    past: List[str] = Field(default_factory=list)
    future: List[str] = Field(default_factory=list)


class LLMTTranslationRequest(BaseModel):
    sentence: str
    target_language: str
    context: Context = Field(default_factory=Context)
    terminology: Optional[List[TermPair]] = None
    similar_translations: Optional[List[SimilarTranslation]] = None


# Backward-compatible alias for existing imports/callers.
LLMTTranslateRequest = LLMTTranslationRequest


class LLMTTranslateResponse(BaseModel):
    translation: str
    model: str
    runtime_ms: float = 0


def build_prompt(request: LLMTTranslationRequest) -> str:
    parts = []
    line_feed = "⏎"

    parts.append(f"TASK:")
    parts.append(f"Translate into {request.target_language}.")
    parts.append(f"")

    if request.terminology:
        parts.append("Preferred terminology:")
        for t in request.terminology:
            parts.append(f"- {t.source} → {t.target}")
        parts.append("")

    if request.similar_translations:
        parts.append("Related translations:")
        for st in request.similar_translations:
            parts.append(f"- SRC: {st.source}")
            parts.append(f"- TGT: {st.target}")
        parts.append("")

    parts.append(f"INSTRUCTIONS:")
    parts.append(f"- Translate ONLY the INPUT section.")
    parts.append(f"- Output a natural and correct sentence.")
    if request.context.past or request.context.future:
        parts.append(f"- Use CONTEXT to resolve meaning (references, agreement).")
        parts.append(f"- Ensure the translation is fully consistent with the CONTEXT.")
        parts.append(f"- Output ONLY the translation.")
    if line_feed in request.sentence:
        parts.append(f"- Preserve all {line_feed} symbols EXACTLY in the same positions as in INPUT.")
    parts.append("")

    if request.context.past:
        parts.append("BEFORE CONTEXT:")
        for sentence in request.context.past:
            parts.append(f"{sentence}")
        parts.append("")

    parts.append("INPUT:")
    parts.append(f"{request.sentence}")
    parts.append("")

    if request.context.future:
        parts.append("AFTER CONTEXT:")
        for sentence in request.context.future:
            parts.append(f"{sentence}")
        parts.append("")

    parts.append("OUTPUT:")
    return "\n".join(parts)


def translate_endpoint(request: LLMTTranslationRequest) -> LLMTTranslateResponse:
    tic = time.perf_counter()
    with model_lock:
        model = model_store["model"]
    logger.info(f"\nREQUEST\n{':'*80}\n{request}\n{':'*80}")

    sentence_for_prompt = request.sentence.replace("\n", "⏎")
    logging.info(f"Sentence for prompt: {sentence_for_prompt}")
    prompt = build_prompt(LLMTTranslationRequest(
        sentence=sentence_for_prompt,
        target_language=request.target_language,
        context=request.context,
        terminology=request.terminology,
        similar_translations=request.similar_translations,
    ))
    try:
        response = ollama.generate(model=model, prompt=prompt)
        translation = (response.get("response") or "")
        translation = translation.strip()
        logger.info(f"\n{'#'*80}\n{prompt}{translation}\n{'#'*80}")
        translation = translation.replace("⏎", "\n")
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}")
    runtime_s = time.perf_counter() - tic
    return LLMTTranslateResponse(
        translation=translation,
        model=model,
        runtime_ms=runtime_s * 1000,
    )

