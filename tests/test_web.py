from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

from ai_rag_demo.web import create_app


def _client() -> TestClient:
    app = create_app(
        embeddings=DeterministicFakeEmbedding(size=16),
        llm=FakeListChatModel(responses=["The balance is $43.00."]),
    )
    return TestClient(app)


class TestIndex:
    def test_serves_html(self) -> None:
        response = _client().get("/")
        assert response.status_code == 200
        assert "AI RAG Demo" in response.text


class TestSample:
    def test_returns_the_bundled_sample_document(self) -> None:
        response = _client().get("/api/sample")
        assert response.status_code == 200
        assert "balance" in response.json()["text"].lower()


class TestIngestAndAsk:
    def test_400_when_asking_before_ingesting(self) -> None:
        response = _client().post("/api/ask", json={"question": "What is the balance?"})
        assert response.status_code == 400

    def test_ingest_then_ask_returns_a_real_answer_and_sources(self) -> None:
        client = _client()
        ingest_response = client.post(
            "/api/ingest",
            json={
                "document_text": (
                    "The account balance is $43.00.\n\nThe next billing date is 2026-09-17."
                )
            },
        )
        assert ingest_response.status_code == 200
        assert ingest_response.json()["chunks"] > 0

        ask_response = client.post("/api/ask", json={"question": "What is the balance?"})
        assert ask_response.status_code == 200
        body = ask_response.json()
        assert body["answer"] == "The balance is $43.00."
        assert len(body["sources"]) > 0

    def test_ingest_with_no_body_uses_the_bundled_sample(self) -> None:
        client = _client()
        response = client.post("/api/ingest", json={})
        assert response.status_code == 200
        assert response.json()["chunks"] > 0
