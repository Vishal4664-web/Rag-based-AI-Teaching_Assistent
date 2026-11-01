from __future__ import annotations

from typing import Any
import logging
import threading

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"title", "number", "start", "end", "text", "embedding"}


class RAGError(RuntimeError):
    pass


class RAGService:
    """Compatible with the existing DataFrame + Joblib + Ollama RAG workflow."""

    def __init__(self) -> None:
        self._data: pd.DataFrame | None = None
        self._matrix: np.ndarray | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._data is not None and self._matrix is not None

    def load_data(self, force: bool = False) -> None:
        with self._lock:
            if self.is_loaded and not force:
                return

            path = settings.EMBEDDINGS_PATH
            if not path.exists():
                raise RAGError(
                    f"Embeddings file not found: {path}. "
                    "Copy your embeddings.joblib into backend/data/."
                )

            loaded: Any = joblib.load(path)
            if not isinstance(loaded, pd.DataFrame):
                raise RAGError(
                    "embeddings.joblib must contain a Pandas DataFrame. "
                    f"Received: {type(loaded).__name__}"
                )

            missing = REQUIRED_COLUMNS.difference(loaded.columns)
            if missing:
                raise RAGError(
                    "Embedding DataFrame is missing columns: "
                    + ", ".join(sorted(missing))
                )

            data = loaded.copy()
            valid_rows = data["embedding"].apply(
                lambda value: isinstance(value, (list, tuple, np.ndarray))
            )
            data = data.loc[valid_rows].copy()
            if data.empty:
                raise RAGError("No valid embedding vectors were found in the DataFrame.")

            try:
                matrix = np.vstack(data["embedding"].apply(
                    lambda value: np.asarray(value, dtype=np.float32)
                ).to_list())
            except (TypeError, ValueError) as exc:
                raise RAGError(f"Could not build embedding matrix: {exc}") from exc

            if matrix.ndim != 2 or matrix.shape[0] != len(data):
                raise RAGError("Embedding vectors have inconsistent dimensions.")
            if not np.isfinite(matrix).all():
                raise RAGError("Embedding vectors contain NaN or infinite values.")

            self._data = data.reset_index(drop=True)
            self._matrix = matrix
            logger.info("Loaded %s transcript chunks with vector size %s",
                        len(data), matrix.shape[1])

    def _post_ollama(self, endpoint: str, payload: dict) -> dict:
        url = f"{settings.OLLAMA_BASE_URL}{endpoint}"
        try:
            response = requests.post(
                url, json=payload, timeout=settings.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise RAGError(
                f"Could not reach Ollama at {url}: {exc}. "
                "Check that Ollama is running and the requested model is installed."
            ) from exc
        except ValueError as exc:
            raise RAGError(f"Ollama returned an invalid JSON response: {exc}") from exc

    def embed_text(self, text: str) -> np.ndarray:
        result = self._post_ollama(
            "/api/embed",
            {"model": settings.EMBEDDING_MODEL, "input": text}
        )
        embeddings = result.get("embeddings")
        if not embeddings or not embeddings[0]:
            raise RAGError("Ollama did not return an embedding for the question.")
        vector = np.asarray(embeddings[0], dtype=np.float32)
        if self._matrix is not None and vector.shape[0] != self._matrix.shape[1]:
            raise RAGError(
                "Question embedding size does not match stored embeddings. "
                "Make sure both were created using the same embedding model."
            )
        return vector

    @staticmethod
    def _safe_value(value: Any) -> Any:
        if pd.isna(value) if not isinstance(value, (list, tuple, np.ndarray)) else False:
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _format_time(value: Any) -> str:
        try:
            seconds = max(0, int(float(value)))
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"
        except (TypeError, ValueError):
            return str(value) if value is not None else ""

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict]:
        self.load_data()
        assert self._data is not None and self._matrix is not None

        query_vector = self.embed_text(question).reshape(1, -1)
        scores = cosine_similarity(query_vector, self._matrix)[0]
        k = max(1, min(top_k or settings.TOP_K, len(scores)))
        indices = np.argsort(scores)[::-1][:k]

        results = []
        for index in indices:
            row = self._data.iloc[int(index)]
            score = float(scores[index])
            if score < settings.SIMILARITY_THRESHOLD:
                continue

            start = self._safe_value(row["start"])
            end = self._safe_value(row["end"])
            results.append({
                "title": str(self._safe_value(row["title"]) or "Untitled lecture"),
                "number": self._safe_value(row["number"]),
                "start": start,
                "end": end,
                "start_time": self._format_time(start),
                "end_time": self._format_time(end),
                "text": str(self._safe_value(row["text"]) or ""),
                "score": round(score, 4),
            })
        return results

    @staticmethod
    def build_prompt(question: str, sources: list[dict]) -> str:
        context = "\n\n".join(
            f"[Source {i+1} | Lecture {s['number']} | {s['title']} | "
            f"{s['start_time']}-{s['end_time']}]\n{s['text']}"
            for i, s in enumerate(sources)
        )
        return f"""You are a helpful AI teaching assistant.
Answer the student's question using ONLY the lecture context below.
Explain concepts clearly and simply. You may organize the answer with short headings or bullets.
Do not invent lecture facts or pretend that information appears in the course when it does not.
If the context does not contain enough information, clearly say that the available lecture material does not provide the answer.
Do not include fake timestamps or sources. The application displays source references separately.

LECTURE CONTEXT:
{context}

STUDENT QUESTION:
{question}

ANSWER:"""

    def generate_answer(self, question: str, sources: list[dict]) -> str:
        prompt = self.build_prompt(question, sources)
        result = self._post_ollama(
            "/api/generate",
            {
                "model": settings.LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2}
            }
        )
        answer = result.get("response", "").strip()
        if not answer:
            raise RAGError("The language model returned an empty answer.")
        return answer

    def ask(self, question: str, top_k: int | None = None) -> dict:
        sources = self.retrieve(question, top_k=top_k)
        if not sources:
            return {
                "answer": (
                    "I couldn't find sufficiently relevant information in the available "
                    "lecture material. Try rephrasing your question or asking about a "
                    "topic covered in the course."
                ),
                "sources": [],
                "retrieved_count": 0
            }
        answer = self.generate_answer(question, sources)
        return {
            "answer": answer,
            "sources": [
                {key: source[key] for key in (
                    "title", "number", "start", "end",
                    "start_time", "end_time", "score"
                )}
                for source in sources
            ],
            "retrieved_count": len(sources)
        }

    def make_quiz(self, topic: str, question_count: int = 5) -> dict:
        sources = self.retrieve(topic, top_k=max(settings.TOP_K, 8))
        if not sources:
            raise RAGError("No relevant lecture content was found for this quiz topic.")
        context = "\n\n".join(source["text"] for source in sources)
        prompt = f"""Create {question_count} multiple-choice questions using ONLY this lecture context.
Return valid JSON only, with this exact shape:
{{"questions":[{{"question":"...","options":["A...","B...","C...","D..."],"answer":"A...","explanation":"..."}}]}}
Do not add markdown fences. Make sure each answer is supported by the context.

LECTURE CONTEXT:
{context}

TOPIC: {topic}
"""
        result = self._post_ollama(
            "/api/generate",
            {
                "model": settings.LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2}
            }
        )
        import json
        try:
            quiz = json.loads(result.get("response", "{}"))
        except json.JSONDecodeError as exc:
            raise RAGError("The model did not return valid quiz JSON. Try again.") from exc
        return {"quiz": quiz, "sources": [
            {key: source[key] for key in ("title", "number", "start_time", "end_time", "score")}
            for source in sources
        ]}


rag_service = RAGService()
