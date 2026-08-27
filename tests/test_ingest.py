from langchain_core.embeddings import DeterministicFakeEmbedding

from ai_rag_demo.ingest import build_vectorstore
from ai_rag_demo.loader import split_text


class TestBuildVectorstore:
    def test_embeds_chunks_and_supports_similarity_search(self) -> None:
        chunks = split_text(
            "The account balance is $43.00.\n\n"
            "The next billing date is 2026-09-17.\n\n"
            "The device on file is a Model X200.",
            source="test",
            chunk_size=40,
            chunk_overlap=0,
        )
        embeddings = DeterministicFakeEmbedding(size=16)
        vectorstore = build_vectorstore(chunks, embeddings, persist_directory=None)

        results = vectorstore.similarity_search("What is the account balance?", k=1)
        assert len(results) == 1
        assert results[0].page_content in {chunk.page_content for chunk in chunks}
