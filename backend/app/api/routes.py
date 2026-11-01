from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.rag_service import rag_service, RAGError

router = APIRouter(prefix="/api")


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    course_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class QuizRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=300)
    question_count: int = Field(default=5, ge=1, le=10)
    course_id: str | None = None


@router.get("/health")
def health():
    try:
        rag_service.load_data()
        data_status = "loaded"
        chunk_count = len(rag_service._data)
    except Exception as exc:
        data_status = f"not ready: {exc}"
        chunk_count = 0

    try:
        import requests
        response = requests.get(
            f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5
        )
        ollama_status = "connected" if response.ok else "unavailable"
    except Exception:
        ollama_status = "unavailable"

    return {
        "status": "ok" if data_status == "loaded" and ollama_status == "connected" else "degraded",
        "data": data_status,
        "chunk_count": chunk_count,
        "ollama": ollama_status,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_model": settings.LLM_MODEL
    }


@router.get("/courses")
def courses():
    return [{
        "id": settings.COURSE_ID,
        "name": settings.COURSE_NAME
    }]


@router.post("/chat")
def chat(request: ChatRequest):
    try:
        return rag_service.ask(request.question, top_k=request.top_k)
    except RAGError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected RAG error: {exc}") from exc


@router.post("/quiz")
def quiz(request: QuizRequest):
    try:
        return rag_service.make_quiz(request.topic, request.question_count)
    except RAGError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {exc}") from exc
