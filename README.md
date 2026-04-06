# LLMT — FastAPI LLM Translation Service

A FastAPI application for sentence translation using Large Language Models served via **ollama**. Supports health monitoring, dynamic model loading, and context-aware translation with optional terminology and fuzzy-match hints.

## Features
- Loads `mistralai/Ministral-3-3B-Instruct-2512-GGUF` as the default model at startup
- `/health`: Returns the currently loaded model and service status
- `/upload`: Pull and activate any model supported by ollama
- `/translate`: Translate a sentence into a target language, with optional context (previous sentence, terminology, similar translations)

## Load `mistralai/Ministral-3-3B-Instruct-2512-GGUF` with Ollama

For Hugging Face-hosted GGUF models, use the `hf.co/` prefix with Ollama.

### Run directly
```bash
ollama run hf.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF
```

### Pull a specific quantization, then run
```bash
ollama pull hf.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF:Q4_K_M
ollama run hf.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF:Q4_K_M
```

### Suggested quantization by available RAM (Mac)
- `Q4_K_M`: best default for most machines (good speed/quality balance)
- `Q5_K_M`: if you have more RAM and want a bit more quality
- `Q8_0`: highest quality, much heavier memory usage

If you are unsure, start with `Q4_K_M`.

## Endpoints

### `GET /health`
Returns the active model name and service status.

### `POST /upload`
Pull a model via ollama and set it as the active translation model.

**Request (JSON):**
```json
{
  "model": "hf.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF:Q4_K_M"
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
- [ollama](https://ollama.com/) installed and running locally (see "Install and Launch Ollama (macOS)" below)

### Install and Launch Ollama (macOS)

1. Install Ollama with Homebrew:
  ```bash
  brew install ollama
  ```

2. Start Ollama as a background service:
  ```bash
  brew services start ollama
  ```

3. Verify the CLI is available:
  ```bash
  ollama --version
  ```

4. Verify the local Ollama API is reachable:
  ```bash
  curl -sS http://localhost:11434/api/tags
  ```

5. Optional: check service status:
  ```bash
  brew services list | grep '^ollama[[:space:]]'
  ```

If Homebrew is not installed, install it first from https://brew.sh.

### Install and Launch Ollama (Linux)

1. Install Ollama:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

2. Start Ollama:
  ```bash
  ollama serve
  ```

3. Verify the local Ollama API is reachable:
  ```bash
  curl -sS http://localhost:11434/api/tags
  ```

If your system uses systemd, you can run Ollama as a background service:

```bash
sudo systemctl enable --now ollama
systemctl status ollama
```

### Troubleshooting Ollama

1. `ollama: command not found`
   - macOS (Homebrew):
     ```bash
     brew install ollama
     ```
   - Linux:
     ```bash
     curl -fsSL https://ollama.com/install.sh | sh
     ```
   - Open a new terminal after installation, then run:
     ```bash
     ollama --version
     ```

2. Port `11434` already in use
   - Find the process using the port:
     ```bash
     lsof -i :11434
     ```
   - Stop the conflicting process or stop existing Ollama instance:
     ```bash
     brew services stop ollama
     ```
     or on Linux:
     ```bash
     sudo systemctl stop ollama
     ```

3. Ollama service fails to start
   - macOS:
     ```bash
     brew services list | grep '^ollama[[:space:]]'
     ```
   - Linux:
     ```bash
     systemctl status ollama
     journalctl -u ollama -n 100 --no-pager
     ```
   - API health check:
     ```bash
     curl -sS http://localhost:11434/api/tags
     ```

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
