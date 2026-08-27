"""Configuration for the LangChain + Ollama + Chroma RAG demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = ["DEFAULT_CONFIG", "RagConfig"]


@dataclass(frozen=True, slots=True)
class RagConfig:
    """Runtime settings for embeddings, generation, and chunking."""

    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text:latest"
    llm_model: str = "llama3.2:latest"
    chunk_size: int = 800
    chunk_overlap: int = 100
    retrieval_k: int = 3
    collection_name: str = "rag_demo"


DEFAULT_CONFIG: Final[RagConfig] = RagConfig()
