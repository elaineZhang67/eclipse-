import os


DEFAULT_MEMORY_DB = os.environ.get(
    "SURVEILLANCE_MEMORY_DB",
    "memory_store/surveillance_memory.db",
)
VIDEO_STORAGE_DIR = os.environ.get(
    "SURVEILLANCE_VIDEO_STORAGE_DIR",
    "backend_storage/videos",
)

DEFAULT_DEVICE = os.environ.get("SURVEILLANCE_DEVICE", "auto")
DEFAULT_QA_BACKEND = os.environ.get("SURVEILLANCE_QA_BACKEND", "text")
DEFAULT_QA_MODEL = os.environ.get("SURVEILLANCE_QA_MODEL", "")
