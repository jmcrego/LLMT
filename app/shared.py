import threading

DEFAULT_MODEL = "gemma3:4b"

# Shared model state for the app
model_store = {"model": DEFAULT_MODEL}
model_lock = threading.Lock()
