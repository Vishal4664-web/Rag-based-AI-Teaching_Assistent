from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings

app = FastAPI(
    title="RAG AI Teaching Assistant API",
    description="Ask questions about lecture transcripts using BGE-M3 retrieval and Llama 3.2.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.FRONTEND_ORIGIN,
        "http://127.0.0.1:5173",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def root():
    return {
        "message": "RAG AI Teaching Assistant API is running",
        "docs": "/docs",
        "health": "/api/health"
    }
