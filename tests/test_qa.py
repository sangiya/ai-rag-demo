from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

from ai_rag_demo.ingest import build_vectorstore
from ai_rag_demo.loader import split_text
from ai_rag_demo.qa import ask, build_qa_chain


class TestBuildQaChain:
    def test_ask_returns_the_llm_answer_and_real_retrieved_sources(self) -> None:
        chunks = split_text(
            "The account balance is $43.00.\n\n"
            "The next billing date is 2026-09-17.\n\n"
            "The device on file is a Model X200.",
            source="test",
            chunk_size=40,
            chunk_overlap=0,
        )
        vectorstore = build_vectorstore(
            chunks, DeterministicFakeEmbedding(size=16), persist_directory=None
        )
        llm = FakeListChatModel(responses=["The balance is $43.00."])
        qa = build_qa_chain(vectorstore, llm, k=2)

        result = ask(qa, "What is the account balance?")

        assert result["answer"] == "The balance is $43.00."
        assert len(result["sources"]) == 2
        retrieved_content = {source["content"] for source in result["sources"]}
        assert retrieved_content <= {chunk.page_content for chunk in chunks}
