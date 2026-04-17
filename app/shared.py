import threading

DEFAULT_MODEL = "gemma3:4b"
DEFAULT_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
DEFAULT_MODEL = "qwen3:4b"
DEFAULT_MODEL = "translategemma:4b"
DEFAULT_MODEL = "qwen3.5:4b"

# Shared model state for the app
model_store = {"model": DEFAULT_MODEL}
model_lock = threading.Lock()
