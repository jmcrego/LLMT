import threading

DEFAULT_MODEL = "mistralai/Ministral-3-3B-Instruct-2512-GGUF"

# Shared model state for the app
model_store = {"model": DEFAULT_MODEL}
model_lock = threading.Lock()
