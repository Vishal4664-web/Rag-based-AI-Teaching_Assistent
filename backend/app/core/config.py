import os
from pathlib import Path
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_DIR / ".env")

def _resolve_data_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = BACKEND_DIR / path
    return path.resolve()

class Settings:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
    EMBEDDINGS_PATH = _resolve_data_path(
        os.getenv("EMBEDDINGS_PATH", "./data/embeddings.joblib")
    )
    TOP_K = int(os.getenv("TOP_K", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.20"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "180"))
    FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
    COURSE_ID = os.getenv("COURSE_ID", "sigma-web-development")
    COURSE_NAME = os.getenv("COURSE_NAME", "Sigma Web Development")

settings = Settings()
