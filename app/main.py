from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

from .health import health_endpoint, LLMTHealthResponse
from .upload import upload_endpoint, LLMTUploadRequest, LLMTUploadResponse
from .translate import translate_endpoint, translate_stream_endpoint, LLMTTranslationRequest, LLMTTranslateResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # marks the point where the app starts accepting requests


# FastAPI app
app = FastAPI(lifespan=lifespan)

# Health check endpoint, returns current model and status
@app.get("/health", response_model=LLMTHealthResponse)
def health():
    return health_endpoint()

# Upload endpoint, pulls and sets the active model via ollama
@app.post("/upload", response_model=LLMTUploadResponse)
def upload(request: LLMTUploadRequest):
    return upload_endpoint(request)

# Translate endpoint, translates a sentence using the loaded model
@app.post("/translate", response_model=LLMTTranslateResponse)
def translate(request: LLMTTranslationRequest):
    return translate_endpoint(request)

# Streaming translate endpoint, streams translation tokens in real-time
@app.post("/translate-stream")
async def translate_stream(request: LLMTTranslationRequest):
    return StreamingResponse(translate_stream_endpoint(request), media_type="application/x-ndjson")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
