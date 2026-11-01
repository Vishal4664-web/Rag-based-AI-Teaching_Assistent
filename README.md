# RAG AI Teaching Assistant

A React + FastAPI web application for asking questions about video-lecture transcripts. It uses your existing RAG approach:

- **BGE-M3 through Ollama** to embed questions.
- **Cosine similarity** to retrieve relevant transcript chunks.
- **Llama 3.2 through Ollama** to generate answers grounded in retrieved lecture content.
- **Joblib + Pandas** to load the existing `embeddings.joblib` DataFrame.

The backend expects the DataFrame columns used by your current project: `title`, `number`, `start`, `end`, `text`, and `embedding`.

> This package provides the new frontend, API layer, and a compatible RAG service. Your original ingestion scripts (`video_to_mp3.py`, `mp3_to_json.py`, `merge_chunks.py`, and `preprocess_json.py`) were not available in this chat, so they are not overwritten or recreated. Copy them into `backend/ingestion/` if you want them in this project.

## Features

- ChatGPT-style responsive chat interface.
- Course selector (initially configured for one course).
- Source cards with lecture title, lecture number, and timestamps.
- Configurable top-K retrieval and similarity threshold.
- Context-grounded answering with a clear fallback when relevant context is not found.
- Health endpoint to check API and Ollama connectivity.
- Chat history stored in the browser's local storage.
- Basic quiz generation endpoint using retrieved lecture context.
- Configurable model names, Ollama URL, and embedding file path.

## Architecture

```text
React + Vite
    |
    | POST /api/chat
    v
FastAPI
    |
    v
RAG service
    |-- Load embeddings.joblib (Pandas DataFrame)
    |-- Embed question with Ollama /api/embed (BGE-M3)
    |-- Retrieve top-K chunks with cosine similarity
    |-- Build context-restricted prompt
    |-- Generate answer with Ollama /api/generate (Llama 3.2)
    v
Answer + lecture source metadata
```

## Project structure

```text
RAG-AI-Teaching-Assistant/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/routes.py
│   │   ├── core/config.py
│   │   └── services/rag_service.py
│   ├── data/
│   │   └── README.md
│   ├── ingestion/             # Put your existing ingestion scripts here
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── services/api.js
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── index.html
│   └── package.json
├── .gitignore
└── README.md
```

## 1. Prerequisites

- Python 3.10 or newer
- Node.js 20 or newer
- Ollama installed and running
- Your existing `embeddings.joblib` file

Pull the models (if they are not already installed):

```bash
ollama pull bge-m3
ollama pull llama3.2
```

Make sure Ollama is running:

```bash
ollama serve
```

If Ollama is already running as a Windows service/application, you do not need to start it again.

## 2. Add your existing embeddings

Copy your existing file:

```text
C:\Rag based ai\embeddings.joblib
```

to:

```text
RAG-AI-Teaching-Assistant/backend/data/embeddings.joblib
```

Do not commit this file to a public GitHub repository if it is large or contains private course material.

## 3. Run the backend

Open a terminal in the project folder:

```powershell
cd backend
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

Install packages and configure settings:

```powershell
pip install -r requirements.txt
copy .env.example .env
```

Run FastAPI:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API documentation:

- http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/api/health

## 4. Run the frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (normally http://localhost:5173).

## 5. Configuration

Edit `backend/.env`:

```env
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=bge-m3
LLM_MODEL=llama3.2
EMBEDDINGS_PATH=./data/embeddings.joblib
TOP_K=5
SIMILARITY_THRESHOLD=0.20
```

The similarity threshold is a starting value, not a universal quality guarantee. Test it against your own lecture questions and tune it. Set it lower to retrieve more loosely related chunks; raise it to be stricter.

## 6. Expected embedding data

The service expects a Pandas DataFrame with these columns:

| Column | Meaning |
|---|---|
| `title` | Lecture/video title |
| `number` | Lecture number |
| `start` | Chunk start time in seconds |
| `end` | Chunk end time in seconds |
| `text` | Transcript text |
| `embedding` | BGE-M3 vector for that chunk |

If your current Joblib file uses different column names or stores vectors in a different structure, adjust the column mapping in `backend/app/services/rag_service.py`.

## 7. API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | API, data, and Ollama status |
| GET | `/api/courses` | Available course list |
| POST | `/api/chat` | Retrieve context and answer a question |
| POST | `/api/quiz` | Generate practice questions from retrieved chunks |

Example request:

```json
{
  "question": "What are semantic HTML tags?",
  "course_id": "sigma-web-development"
}
```

## 8. Performance and quality notes

- The embeddings are loaded once when the backend starts rather than on every chat request.
- Query embeddings are created only when a question is submitted.
- Retrieval uses a NumPy matrix and scikit-learn cosine similarity.
- Top-K and similarity threshold are configurable.
- The prompt tells the model to use only the supplied lecture context and to say when the context does not contain the answer.
- The first model call can be slower while Ollama loads the model into memory.
- For a much larger transcript collection, consider FAISS or ChromaDB. This package intentionally keeps your current Joblib + DataFrame workflow to avoid forcing a migration.

## 9. Important limitations

- Chat history is currently saved in the user's browser only; it is not synchronized across devices.
- The course selector is a UI/API foundation. Add separate datasets and course metadata before enabling multiple real courses.
- Quiz generation is a basic LLM feature and should be checked for correctness.
- The backend requires Ollama to be reachable from the machine running FastAPI.
- Do not expose Ollama or the API publicly without authentication, rate limiting, and suitable network security.

## Troubleshooting

**`embeddings.joblib` not found**
- Put the file in `backend/data/`, or update `EMBEDDINGS_PATH` in `.env`.

**Ollama connection error**
- Confirm Ollama is running at `http://localhost:11434`.
- Test `http://localhost:11434/api/tags` in a browser or with curl.

**Embedding model not found**
```bash
ollama pull bge-m3
```

**LLM model not found**
```bash
ollama pull llama3.2
```

**No relevant lecture context found**
- Check that your embeddings were generated using the same embedding model (`bge-m3`).
- Check that the DataFrame contains the expected columns.
- Test a question that closely matches your lecture transcript.
- Tune `SIMILARITY_THRESHOLD` after evaluating several questions.

## License and course content

Use only lecture recordings and transcripts that you are permitted to process and share.
