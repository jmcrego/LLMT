# LLMT — FastAPI LLM Translation Service

A FastAPI application for sentence translation using Large Language Models served via **ollama**. Supports health monitoring, dynamic model loading, and context-aware translation with optional terminology and fuzzy-match hints.

## Features
- Loads `mistralai/Ministral-3-3B-Instruct-2512-GGUF` as the default model at startup
- `/health`: Returns the currently loaded model and service status
- `/upload`: Pull and activate any model supported by ollama
- `/translate`: Translate a sentence into a target language, with optional context (previous sentence, terminology, similar translations)

## Endpoints

### `GET /health`
Returns the active model name and service status.

### `POST /upload`
Pull a model via ollama and set it as the active translation model.

**Request (JSON):**
```json
{
  "model": "mistralai/Ministral-3-3B-Instruct-2512-GGUF"
}
```

### `POST /translate`
Translate a sentence using the active model.

**Request (JSON):**
```json
{
  "sentence": "The cat sat on the mat.",
  "target_language": "French",
  "previous_sentence": "It was a sunny day.",
  "terminology": [
    {"source": "cat", "target": "chat"}
  ],
  "similar_translations": [
    {"source": "The dog sat on the mat.", "target": "Le chien était assis sur le tapis."}
  ]
}
```

Fields `previous_sentence`, `terminology`, and `similar_translations` are optional.

## Setup

### Prerequisites
- [ollama](https://ollama.com/) installed and running locally (`ollama serve`)

### Installation

1. Create a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install requirements:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Running the Server

```bash
uvicorn app.main:app --reload --port 8003
```

## Example Requests

### Health Check
```bash
curl http://localhost:8003/health
```

### Load a Model
```bash
curl -X POST "http://localhost:8003/upload" \
  -H "Content-Type: application/json" \
  -d '{"model": "mistral"}'
```

### Translate a Sentence
```bash
curl -X POST "http://localhost:8003/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "sentence": "The cat sat on the mat.",
    "target_language": "French"
  }'
```

### Translate with Context
```bash
curl -X POST "http://localhost:8003/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "sentence": "The cat sat on the mat.",
    "target_language": "French",
    "previous_sentence": "It was a sunny day.",
    "terminology": [{"source": "cat", "target": "chat"}],
    "similar_translations": [
      {"source": "The dog sat on the mat.", "target": "Le chien était assis sur le tapis."}
    ]
  }'
```

## Directory Structure
- `app/main.py` — FastAPI application and route definitions
- `app/shared.py` — Shared model state and threading lock
- `app/health.py` — Health endpoint
- `app/upload.py` — Upload/model-load endpoint
- `app/translate.py` — Translation endpoint and prompt builder

## License
MIT
