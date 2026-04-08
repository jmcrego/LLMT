import time
import ollama
import logging

from fastapi import HTTPException
from pathlib import Path
from pydantic import BaseModel
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
    parts.append(
        f"TASK:\n"
        f"Translate the text between [START] and [END] into {request.target_language}.\n"
        f"\n"
    )

    if request.previous_sentence:
        parts.append("Context:")
        parts.append(request.previous_sentence)
        parts.append("")

    if request.terminology:
        parts.append("Use the following terminology when translating:")
        for t in request.terminology:
            parts.append(f" - {t.source} → {t.target}")
        parts.append("")

    if request.similar_translations:
        parts.append(f"Similar translations:")
        for st in request.similar_translations:
            parts.append(
                f"- SRC: {st.source}\n"
                f"- TGT: {st.target}\n"
                f"\n"
            )
    parts.append(
            f"RULES:\n"
            f"- Output only the translation text, with no prefix, suffix, quotes, or explanations.\n"
            f"- If the input contain symbols ⏎. the output must also insert them in the same positions.\n"
            f"\n"
            f"[START]\n"
            f"{request.sentence}\n"
            f"[END]\n"
            f"\n"
            f"Output:\n"
    )
    return "".join(parts)


def translate_endpoint(request: LLMTTranslateRequest) -> LLMTTranslateResponse:
    tic = time.perf_counter()
    with model_lock:
        model = model_store["model"]
    sentence_for_prompt = request.sentence.replace("\n", "⏎")
    prompt = build_prompt(LLMTTranslateRequest(
        sentence=sentence_for_prompt,
        target_language=request.target_language,
        previous_sentence=request.previous_sentence,
        terminology=request.terminology,
        similar_translations=request.similar_translations,
    ))
    logger.info(f"\n{'='*50}\nPROMPT\n{prompt}\n{'='*50}")
    try:
        response = ollama.generate(model=model, prompt=prompt)
        translation = (response.get("response") or "")
        logger.info(f"\n{'-'*50}\nTRANSLATION\n{translation}\n{'-'*50}")
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

