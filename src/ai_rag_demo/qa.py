"""Build a retrieval-augmented question-answering chain and ask it questions."""

from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from .config import DEFAULT_CONFIG, RagConfig

__all__ = ["ask", "build_qa_chain", "default_llm"]


def default_llm(config: RagConfig = DEFAULT_CONFIG) -> ChatOllama:
    """Build the real chat model, which talks to a local Ollama server."""
    return ChatOllama(model=config.llm_model, base_url=config.ollama_base_url, temperature=0)


def build_qa_chain(
    vectorstore: Chroma, llm: BaseChatModel, k: int = DEFAULT_CONFIG.retrieval_k
) -> RetrievalQA:
    """Wire a retriever over ``vectorstore`` into a "stuff"-style QA chain."""
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        return_source_documents=True,
    )


def ask(qa: RetrievalQA, question: str) -> dict[str, Any]:
    """Run ``question`` through ``qa`` and return the answer with its sources."""
    result = qa.invoke({"query": question})
    sources = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in result["source_documents"]
    ]
    return {"answer": result["result"], "sources": sources}
