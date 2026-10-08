"""API tests for Exercise 1 with Ollama replaced by a deterministic fake."""

from fastapi.testclient import TestClient

from app.main import app, get_ollama_client, get_rag_service
from knowledge.models import RetrievedChunk


class FakeOllamaClient:
    model = "codellama-test"

    async def generate(self, prompt: str) -> str:
        return f"Test response for: {prompt}"


def override_ollama_client() -> FakeOllamaClient:
    return FakeOllamaClient()


class FakeRagService:
    llm = FakeOllamaClient()

    async def answer(self, question: str) -> tuple[str, list[RetrievedChunk]]:
        source = RetrievedChunk(
            text="Assignment 2 is due at 5:00 PM on 15 October.",
            source="syllabus.md",
            page=None,
            section="Assessment schedule",
            citation="syllabus.md, section: Assessment schedule",
            distance=0.1,
        )
        return (
            "Assignment 2 is due at 5:00 PM on 15 October "
            "[syllabus.md, section: Assessment schedule].",
            [source],
        )


def override_rag_service() -> FakeRagService:
    return FakeRagService()


app.dependency_overrides[get_ollama_client] = override_ollama_client
app.dependency_overrides[get_rag_service] = override_rag_service
client = TestClient(app)


def test_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["exercise"] == "1 - Basic LLM Application"


def test_ask_returns_llm_response() -> None:
    response = client.post("/ask", json={"question": "What is recursion?"})
    assert response.status_code == 200
    assert response.json() == {
        "answer": "Test response for: What is recursion?",
        "model": "codellama-test",
        "exercise": 1,
    }


def test_ask_rejects_empty_question() -> None:
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_rag_answer_contains_traceable_source() -> None:
    response = client.post("/ask/rag", json={"question": "When is assignment 2 due?"})
    assert response.status_code == 200
    body = response.json()
    assert body["exercise"] == 3
    assert "[syllabus.md, section: Assessment schedule]" in body["answer"]
    assert body["sources"][0]["citation"] == (
        "syllabus.md, section: Assessment schedule"
    )
    assert body["sources"][0]["relevance_score"] == 0.95


def test_comparison_returns_same_question_with_both_answers() -> None:
    question = "When is assignment 2 due?"
    response = client.post("/compare", json={"question": question})
    assert response.status_code == 200
    body = response.json()
    assert body["question"] == question
    assert body["without_rag"]["answer"] == f"Test response for: {question}"
    assert body["with_rag"]["sources"][0]["source"] == "syllabus.md"
