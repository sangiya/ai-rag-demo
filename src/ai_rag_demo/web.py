"""A local, single-user web UI over the ingest -> retrieve -> answer pipeline.

No auth, no persistence beyond the current server process's in-memory Chroma
collection, no multi-tenancy -- a developer/learning tool run on one machine.
FastAPI serves both the JSON endpoints and a single static HTML/JS page
directly.

Talks to a real local Ollama server by default (embeddings + chat model), so
``ai-rag-demo serve`` needs Ollama running with the models named in
``config.DEFAULT_CONFIG``. ``create_app`` accepts an injected embedder and
chat model so it can be exercised in tests without a live Ollama server --
see ``tests/test_web.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from .config import DEFAULT_CONFIG
from .ingest import build_vectorstore, default_embeddings
from .loader import split_text
from .qa import ask, build_qa_chain, default_llm

__all__ = ["create_app"]

_SAMPLE_TEXT = (Path(__file__).parent / "sample_data" / "account.txt").read_text(encoding="utf-8")


class IngestRequest(BaseModel):
    document_text: str | None = None


class AskRequest(BaseModel):
    question: str
    k: int = DEFAULT_CONFIG.retrieval_k


class _Session:
    """Holds the current in-memory vector store for one running server."""

    def __init__(self) -> None:
        self.vectorstore: Chroma | None = None
        self.chunk_count: int = 0


_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AI RAG Demo</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #18181b; }
  h1 { font-size: 1.25rem; }
  h2 { font-size: 1rem; margin-top: 2.5rem; border-top: 1px solid #e4e4e7; padding-top: 1.5rem; }
  textarea, input { font-size: 0.85rem; padding: 0.4rem; border: 1px solid #d4d4d8; border-radius: 6px; width: 100%; box-sizing: border-box; margin-top: 0.3rem; font-family: inherit; }
  textarea { min-height: 160px; }
  button { background: #18181b; color: white; border: none; border-radius: 6px; padding: 0.4rem 0.9rem; cursor: pointer; font-size: 0.85rem; margin-top: 0.6rem; }
  button:hover { background: #3f3f46; }
  .card { border: 1px solid #e4e4e7; border-radius: 8px; padding: 0.75rem 1rem; margin-top: 0.75rem; font-size: 0.85rem; }
  pre { font-size: 0.78rem; background: #fafafa; border: 1px solid #e4e4e7; border-radius: 6px; padding: 0.6rem; overflow-x: auto; white-space: pre-wrap; }
  #status { font-size: 0.8rem; color: #71717a; margin-top: 0.4rem; }
</style>
</head>
<body>
<h1>AI RAG Demo</h1>
<p style="font-size:0.8rem;color:#71717a;">Needs a local Ollama server running the embedding and chat models named in the CLI's config -- this is a real retrieval-augmented pipeline, not a canned response.</p>

<h2>1. Ingest a document</h2>
<textarea id="document-text"></textarea>
<button onclick="ingest()">Ingest</button>
<div id="status"></div>

<h2>2. Ask a question</h2>
<input id="question" value="What is the current account balance and when is the next billing date?">
<button onclick="askQuestion()">Ask</button>
<pre id="answer"></pre>

<script>
async function get(url) {
  const res = await fetch(url);
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'request failed');
  return data;
}

async function post(url, body) {
  const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'request failed');
  return data;
}

async function loadSample() {
  const data = await get('/api/sample');
  document.getElementById('document-text').value = data.text;
}

async function ingest() {
  const document_text = document.getElementById('document-text').value;
  document.getElementById('status').textContent = 'Ingesting...';
  try {
    const data = await post('/api/ingest', { document_text });
    document.getElementById('status').textContent = `Ingested ${data.chunks} chunk(s).`;
  } catch (err) {
    document.getElementById('status').textContent = `Error: ${err.message}`;
  }
}

async function askQuestion() {
  const question = document.getElementById('question').value;
  const result = document.getElementById('answer');
  result.textContent = 'Thinking...';
  try {
    const data = await post('/api/ask', { question });
    let text = `Answer: ${data.answer}\\n\\nSources:\\n`;
    data.sources.forEach((s, i) => { text += `--- Source ${i + 1} ---\\n${s.content}\\n\\n`; });
    result.textContent = text;
  } catch (err) {
    result.textContent = `Error: ${err.message}`;
  }
}

loadSample();
</script>
</body>
</html>
"""


def create_app(embeddings: Embeddings | None = None, llm: BaseChatModel | None = None) -> FastAPI:
    app = FastAPI(title="AI RAG Demo")
    session = _Session()
    active_embeddings = embeddings if embeddings is not None else default_embeddings()
    active_llm = llm if llm is not None else default_llm()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _PAGE

    @app.get("/api/sample")
    def sample() -> dict[str, str]:
        return {"text": _SAMPLE_TEXT}

    @app.post("/api/ingest")
    def ingest(request: IngestRequest) -> dict[str, int]:
        text = request.document_text or _SAMPLE_TEXT
        chunks = split_text(text, source="web-ui")
        session.vectorstore = build_vectorstore(
            chunks, active_embeddings, persist_directory=None, collection_name="web-demo"
        )
        session.chunk_count = len(chunks)
        return {"chunks": session.chunk_count}

    @app.post("/api/ask")
    def ask_endpoint(request: AskRequest) -> dict[str, Any]:
        if session.vectorstore is None:
            raise HTTPException(status_code=400, detail="ingest a document before asking")
        qa = build_qa_chain(session.vectorstore, active_llm, k=request.k)
        return ask(qa, request.question)

    return app
