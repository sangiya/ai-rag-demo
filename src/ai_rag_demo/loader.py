"""Load a text document and split it into overlapping chunks.

Reads the file directly rather than going through langchain_community's
``TextLoader`` -- one fewer (and now-deprecated) dependency for what is
just reading a UTF-8 file into a single ``Document``.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import DEFAULT_CONFIG

__all__ = ["load_and_split", "split_text"]


def split_text(
    text: str,
    source: str = "inline",
    chunk_size: int = DEFAULT_CONFIG.chunk_size,
    chunk_overlap: int = DEFAULT_CONFIG.chunk_overlap,
) -> list[Document]:
    """Split raw text into overlapping chunks, tagging each with ``source``."""
    document = Document(page_content=text, metadata={"source": source})
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents([document])


def load_and_split(
    document_path: str | Path,
    chunk_size: int = DEFAULT_CONFIG.chunk_size,
    chunk_overlap: int = DEFAULT_CONFIG.chunk_overlap,
) -> list[Document]:
    """Read ``document_path`` as UTF-8 text and split it into overlapping chunks."""
    path = Path(document_path)
    text = path.read_text(encoding="utf-8")
    return split_text(text, source=str(path), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
