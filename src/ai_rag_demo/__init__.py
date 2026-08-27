"""A practical LangChain + Ollama + Chroma retrieval-augmented generation demo."""

from .config import DEFAULT_CONFIG, RagConfig
from .ingest import build_vectorstore, default_embeddings
from .loader import load_and_split, split_text
from .qa import ask, build_qa_chain, default_llm

__all__ = [
    "DEFAULT_CONFIG",
    "RagConfig",
    "ask",
    "build_qa_chain",
    "build_vectorstore",
    "default_embeddings",
    "default_llm",
    "load_and_split",
    "split_text",
]
