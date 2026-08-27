"""Build a Chroma vector store from document chunks.

The embedder is injected rather than constructed inside ``build_vectorstore``,
so callers (and tests) can substitute a deterministic offline embedder instead
of a real call to a local Ollama server.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from .config import DEFAULT_CONFIG, RagConfig

__all__ = ["build_vectorstore", "default_embeddings"]


def default_embeddings(config: RagConfig = DEFAULT_CONFIG) -> OllamaEmbeddings:
    """Build the real embedder, which talks to a local Ollama server."""
    return OllamaEmbeddings(model=config.embedding_model, base_url=config.ollama_base_url)


def build_vectorstore(
    chunks: list[Document],
    embeddings: Embeddings,
    persist_directory: str | None = None,
    collection_name: str = DEFAULT_CONFIG.collection_name,
) -> Chroma:
    """Embed ``chunks`` and load them into a Chroma collection.

    ``persist_directory`` of ``None`` keeps the collection in memory, which
    is what the web UI and ``demo`` CLI command use so nothing is written
    to disk for a throwaway request.
    """
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
