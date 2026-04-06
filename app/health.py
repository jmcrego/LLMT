from pydantic import BaseModel
from .shared import model_store, model_lock


class LLMTHealthResponse(BaseModel):
    model: str
    status: str


def health_endpoint() -> LLMTHealthResponse:
    with model_lock:
        return LLMTHealthResponse(model=model_store["model"], status="ok")
