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

logger = logging.getLogger("translate.py")

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
    fidelity: Optional[str] = "balanced"  # "strict", "balanced", "fluent"
    formality: Optional[str] = "neutral"  # "neutral", "formal", "informal"
    source_language: Optional[str] = None  # For future use, currently not used in prompt construction.


# Backward-compatible alias for existing imports/callers.
LLMTTranslateRequest = LLMTTranslationRequest


class LLMTTranslateResponse(BaseModel):
    translation: str
    model: str
    runtime_ms: float = 0


def build_prompt_translate(request: LLMTTranslationRequest) -> str:
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
    parts.append(f"- Preserve the original meaning and intent.")
    parts.append(f"- Do not add new information or omit any content.")
    if line_feed in request.sentence:
        parts.append(f"- Preserve all {line_feed} symbols EXACTLY in the same positions as in INPUT.")

    if request.fidelity.lower() == "strict":
        parts.append("- Follow the original wording and structure of the original sentence as closely as possible.")
    elif request.fidelity.lower() == "balanced":
        parts.append("- Translate with the MINIMUM necessary changes to improve clarity and fluency.")
    elif request.fidelity.lower() == "fluent":
        parts.append("- Translate freely to produce a fluent and natural sentence while maintaining the original meaning.")

    if request.formality.lower() == "neutral":
        parts.append("- Follow the formality level of the original sentence, Do not make it more formal or informal.")
    elif request.formality.lower() == "formal":
        parts.append("- Use a formal and professional tone. Prefer precise vocabulary and complete sentence structures. Avoid slang, or colloquial expressions.")
    elif request.formality.lower() == "informal":
        parts.append("- Use a natural and conversational (informal) tone. Avoid overly formal phrasing.")

    parts.append(f"- Make sure you ALWAYS output a grammatical correct {request.target_language} sentence.")
    parts.append("")

    if request.context.past or request.context.future:
        parts.append(f"- Use CONTEXT to resolve meaning (references, agreement).")
        parts.append(f"- Ensure the translated text is fully consistent with the CONTEXT.")
        parts.append(f"- Output ONLY the translated text of the INPUT section.")
        if request.context.past:
            parts.append("BEFORE CONTEXT:")
            for sentence in request.context.past:
                parts.append(f"{sentence}")
        if request.context.future:
            parts.append("AFTER CONTEXT:")
            for sentence in request.context.future:
                parts.append(f"{sentence}")
        parts.append("")

    parts.append("INPUT:")
    parts.append(f"{request.sentence}")
    parts.append("")

    parts.append("OUTPUT:")
    parts.append("")
    return "\n".join(parts)

def build_prompt_revise(request: LLMTTranslationRequest) -> str:
    parts = []
    line_feed = "⏎"

    print(f"Request: {request}")

    parts.append(f"TASK:")
    parts.append(f"Revise the {request.source_language} text.")
    parts.append(f"")
    parts.append(f"INSTRUCTIONS:")
    parts.append(f"- Revise ONLY the INPUT section.")
    parts.append(f"- Preserve the original meaning and intent.")
    parts.append(f"- Do not add new information or omit any content.")
    if line_feed in request.sentence:
        parts.append(f"- Preserve all {line_feed} symbols EXACTLY in the same positions as in INPUT.")

    if request.fidelity.lower() == "strict":
        parts.append("- Introduce the MINIMUM necessary changes to correct grammatical, spelling, or punctuation errors. Modify ONLY errors, nothing else.")
    elif request.fidelity.lower() == "balanced":
        parts.append("- Introduce the MINIMUM necessary wording or structural changes to improve clarity and fluency. Avoid unnecessary rephrasing.")
    elif request.fidelity.lower() == "fluent":
        parts.append("- Rephrase freely to produce a fluent and natural sentence.")

    if request.formality.lower() == "neutral":
        parts.append("- Follow the formality level of the original sentence, Do not make it more formal or informal.")
    elif request.formality.lower() == "formal":
        parts.append("- Use a formal and professional tone. Prefer precise vocabulary and complete sentence structures. Avoid slang, or colloquial expressions.")
    elif request.formality.lower() == "informal":
        parts.append("- Use a natural and conversational (informal) tone. Avoid overly formal phrasing.")

    parts.append(f"- Make sure you ALWAYS output a grammatical correct {request.target_language} sentence.")
    parts.append("")

    if request.context.past or request.context.future:
        parts.append(f"- Use CONTEXT to resolve meaning (references, agreement).")
        parts.append(f"- Ensure the revised text is fully consistent with the CONTEXT.")
        parts.append(f"- Output ONLY the revised text of the INPUT section.")
        if request.context.past:
            parts.append("BEFORE CONTEXT:")
            for sentence in request.context.past:
                parts.append(f"{sentence}")
        if request.context.future:
            parts.append("AFTER CONTEXT:")
            for sentence in request.context.future:
                parts.append(f"{sentence}")
        parts.append("")

    parts.append("INPUT:")
    parts.append(f"{request.sentence}")
    parts.append("")

    parts.append("OUTPUT:")
    parts.append("")

    return "\n".join(parts)

def translate_endpoint(request: LLMTTranslationRequest) -> LLMTTranslateResponse:
    tic = time.perf_counter()
    with model_lock:
        model = model_store["model"]
    logger.info(f"\nREQUEST\n{':'*80}\n{request}\n{':'*80}")

    sentence_for_prompt = request.sentence.replace("\n", "⏎")
    if request.source_language != request.target_language:
        prompt = build_prompt_translate(LLMTTranslationRequest(
            sentence=sentence_for_prompt,
            target_language=request.target_language,            
            context=request.context,
            terminology=request.terminology,
            similar_translations=request.similar_translations,
            fidelity=request.fidelity,
            formality=request.formality,
            source_language=request.source_language,
        ))
    else:
        prompt = build_prompt_revise(LLMTTranslationRequest(
            sentence=sentence_for_prompt,
            target_language=request.target_language,
            context=request.context,
            terminology=request.terminology,
            similar_translations=request.similar_translations,
            fidelity=request.fidelity,
            formality=request.formality,
            source_language=request.source_language,
        ))

    try:
        #curl http://localhost:11434/api/chat -d '{"model": "qwen3.5:4b", "think": false, "stream": false, "messages": [ { "role": "user", "content": "Translate from English into French: Who are you working with?"  } ]}'
        options={
            "temperature": 0.0,     # deterministic
            "top_p": 1.0,
            "top_k": 0,
        }
        logger.info(f"Sending to model: {model} with options {options}")
        response = ollama.chat(
            model=model, 
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            think=False,
            options=options
        )
        if hasattr(response, 'message'):
            translation = (response.message.content.strip() or "")
        else:
            translation = ""
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

