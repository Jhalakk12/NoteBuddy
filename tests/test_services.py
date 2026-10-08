"""Exercise 4 service contracts and orchestration tests."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.ollama_client import GenerationResult, OllamaClient
from knowledge.models import RetrievedChunk
from services.application.main import app as application_app
from services.application.main import get_orchestrator, get_service_client
from services.application.orchestrator import Orchestrator
from services.contracts import (
    DocumentDeleteResponse,
    GenerateResponse,
    DocumentInfo,
    IngestResponse,
    KnowledgeStatus,
    OrchestratedAnswer,
    RetrieveResponse,
    ServiceStatus,
    SourceDto,
    UploadResponse,
)
from services.data.main import app as data_app
from services.data.main import get_document_path, get_store as get_data_store
from services.llm.main import app as llm_app
from services.llm.main import get_llm
from services.retrieval.main import app as retrieval_app
from services.retrieval.main import get_store as get_retrieval_store


class FakeLlm(OllamaClient):
    def __init__(self) -> None:
        self.model = "codellama-test"

    async def generate(self, prompt: str) -> str:
        return "Generated answer"

    async def generate_detailed(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 256,
    ) -> GenerationResult:
        return GenerationResult(answer="Generated answer", model=model or self.model)


def test_llm_service_contract() -> None:
    llm_app.dependency_overrides[get_llm] = FakeLlm
    client = TestClient(llm_app)
    response = client.post("/generate", json={"prompt": "Explain recursion"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Generated answer"
    assert response.json()["model"] == "codellama"
    assert response.json()["prompt_tokens"] is None
    selected = client.post(
        "/generate",
        json={"prompt": "Explain recursion", "model": "qwen2.5-coder:1.5b"},
    )
    assert selected.status_code == 200
    assert selected.json()["model"] == "qwen2.5-coder:1.5b"
    llm_app.dependency_overrides.clear()


class FakeRetrievalStore:
    def query(self, question: str, top_k: int) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                text="Assignment 2 is due Friday.",
                source="syllabus.md",
                page=None,
                section="Assessment",
                citation="syllabus.md, section: Assessment",
                distance=0.1,
            )
        ]


def test_retrieval_service_returns_prompt_and_sources() -> None:
    retrieval_app.dependency_overrides[get_retrieval_store] = FakeRetrievalStore
    client = TestClient(retrieval_app)
    response = client.post(
        "/retrieve", json={"question": "When is assignment 2 due?", "top_k": 3}
    )
    assert response.status_code == 200
    body = response.json()
    assert "using ONLY the supplied course context" in body["grounded_prompt"]
    assert body["sources"][0]["citation"] == "syllabus.md, section: Assessment"
    retrieval_app.dependency_overrides.clear()


class FakeEmbedder:
    model = "fake-embed"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(len(text))] for text in texts]


def test_data_service_ingests_configured_directory(tmp_path: Path) -> None:
    from knowledge.store import KnowledgeStore

    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "syllabus.txt").write_text("Course fact.", encoding="utf-8")
    store = KnowledgeStore(tmp_path / "chroma", FakeEmbedder())
    data_app.dependency_overrides[get_data_store] = lambda: store
    data_app.dependency_overrides[get_document_path] = lambda: documents
    client = TestClient(data_app)
    response = client.post("/ingest", json={"reset": True})
    assert response.status_code == 200
    assert response.json()["collection_total"] == 1
    assert response.json()["embedding_model"] == "fake-embed"
    data_app.dependency_overrides.clear()


def test_data_service_uploads_supported_document(tmp_path: Path) -> None:
    from knowledge.store import KnowledgeStore

    documents = tmp_path / "documents"
    store = KnowledgeStore(tmp_path / "chroma", FakeEmbedder())
    data_app.dependency_overrides[get_data_store] = lambda: store
    data_app.dependency_overrides[get_document_path] = lambda: documents
    client = TestClient(data_app)

    upload = client.post(
        "/documents/upload",
        files={"files": ("lecture.md", b"# Week 1\nEmbeddings", "text/markdown")},
    )
    status = client.get("/documents")

    assert upload.status_code == 200
    assert upload.json()["uploaded"][0]["name"] == "lecture.md"
    assert status.json()["document_count"] == 1
    assert (documents / "lecture.md").read_text() == "# Week 1\nEmbeddings"
    data_app.dependency_overrides.clear()


def test_data_service_deletes_only_selected_document_and_vectors(tmp_path: Path) -> None:
    from knowledge.store import KnowledgeStore

    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "keep.md").write_text("# Keep\nGrounded fact", encoding="utf-8")
    (documents / "remove.md").write_text("# Remove\nOld fact", encoding="utf-8")
    store = KnowledgeStore(tmp_path / "chroma", FakeEmbedder())
    data_app.dependency_overrides[get_data_store] = lambda: store
    data_app.dependency_overrides[get_document_path] = lambda: documents
    client = TestClient(data_app)
    client.post("/ingest", json={"reset": True})

    response = client.delete("/documents/remove.md")

    assert response.status_code == 200
    assert response.json()["deleted"]["name"] == "remove.md"
    assert response.json()["remaining_documents"] == 1
    assert not (documents / "remove.md").exists()
    assert store.count() == 1
    assert client.delete("/documents/../keep.md").status_code in {404, 422}
    data_app.dependency_overrides.clear()


class FakeServiceClient:
    async def retrieve_repository(self, question: str, top_k: int = 4) -> RetrieveResponse:
        return await self.retrieve(question, top_k)

    async def retrieve(self, question: str, top_k: int = 4) -> RetrieveResponse:
        return RetrieveResponse(
            question=question,
            grounded_prompt="Grounded prompt",
            sources=[
                SourceDto(
                    citation="syllabus.md, section: Assessment",
                    source="syllabus.md",
                    page=None,
                    section="Assessment",
                    excerpt="Assignment 2 is due Friday.",
                    relevance_score=0.95,
                )
            ],
        )

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 256,
    ) -> GenerateResponse:
        return GenerateResponse(
            answer="Assignment 2 is due Friday.",
            model=model or "test",
            prompt_tokens=12,
            completion_tokens=8,
            model_memory_bytes=1024**2,
            gpu_memory_bytes=0,
        )

    async def knowledge_status(self) -> KnowledgeStatus:
        return KnowledgeStatus(
            documents=[DocumentInfo(name="syllabus.md", extension=".md", size_bytes=120)],
            document_count=1,
            collection_total=2,
            supported_extensions=[".md", ".pdf", ".txt"],
        )

    async def upload_documents(self, files: list[tuple[str, bytes, str]]) -> UploadResponse:
        return UploadResponse(
            uploaded=[DocumentInfo(name=files[0][0], extension=".md", size_bytes=len(files[0][1]))],
            rejected=[],
        )

    async def delete_document(self, filename: str) -> DocumentDeleteResponse:
        return DocumentDeleteResponse(
            deleted=DocumentInfo(name=filename, extension=Path(filename).suffix, size_bytes=10),
            remaining_documents=0,
            collection_total=0,
        )

    async def ingest(self, reset: bool = False) -> IngestResponse:
        return IngestResponse(
            documents=1,
            units=1,
            chunks=2,
            collection_total=2,
            embedding_model="fake-embed",
        )

    async def service_status(self, name: str, url: str) -> ServiceStatus:
        return ServiceStatus(name=name, url=url, status="ok")

    retrieval_url = "http://retrieval"
    llm_url = "http://llm"
    data_url = "http://data"


def fake_orchestrator() -> Orchestrator:
    return Orchestrator(FakeServiceClient())  # type: ignore[arg-type]


def test_application_service_orchestrates_and_guarantees_citation() -> None:
    application_app.dependency_overrides[get_orchestrator] = fake_orchestrator
    client = TestClient(application_app)
    response = client.post("/ask", json={"question": "When is assignment 2 due?"})
    assert response.status_code == 200
    body = response.json()
    assert body["orchestration"] == ["application", "retrieval", "llm"]
    assert body["answer"].endswith("[syllabus.md, section: Assessment]")
    application_app.dependency_overrides.clear()


def test_application_gateway_exposes_knowledge_and_system_workflows() -> None:
    application_app.dependency_overrides[get_service_client] = FakeServiceClient
    client = TestClient(application_app)

    system = client.get("/system/status")
    knowledge = client.get("/knowledge/status")
    upload = client.post(
        "/knowledge/upload",
        files={"files": ("new-notes.md", b"# Notes", "text/markdown")},
    )
    ingestion = client.post("/knowledge/ingest", json={"reset": True})
    deletion = client.delete("/knowledge/documents/new-notes.md")

    assert system.status_code == 200
    assert system.json()["overall"] == "ok"
    assert len(system.json()["services"]) == 4
    assert knowledge.json()["documents"][0]["name"] == "syllabus.md"
    assert upload.json()["uploaded"][0]["name"] == "new-notes.md"
    assert ingestion.json()["chunks"] == 2
    assert deletion.json()["deleted"]["name"] == "new-notes.md"
    application_app.dependency_overrides.clear()


def test_evaluation_live_comparison_uses_selected_models_and_shared_context() -> None:
    application_app.dependency_overrides[get_service_client] = FakeServiceClient
    client = TestClient(application_app)

    response = client.post(
        "/evaluation/compare-models",
        json={
            "question": "When is Assignment 2 due?",
            "models": ["codellama", "starcoder2:3b"],
            "top_k": 4,
            "max_tokens": 160,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert [row["model"] for row in body["results"]] == ["codellama", "starcoder2:3b"]
    assert all(row["total_tokens"] == 20 for row in body["results"])
    assert body["conditions"]["same_retrieved_context"] is True
    assert len(body["sources"]) == 1
    application_app.dependency_overrides.clear()


def test_live_comparison_routes_code_and_repository_tasks() -> None:
    from week4.evaluator import load_dataset
    application_app.dependency_overrides[get_service_client] = FakeServiceClient
    try:
        client = TestClient(application_app)
        for domain in ('code', 'repository'):
            task = next(item for item in load_dataset() if item['domain'] == domain)
            response = client.post('/evaluation/compare-models', json={
                'question': task['question'], 'domain': domain,
                'models': ['codellama', 'starcoder2:3b'],
            })
            assert response.status_code == 200
            body = response.json()
            assert body['conditions']['domain'] == domain
            assert 'accuracy' in body['results'][0]['metrics']
            if domain == 'code':
                assert body['sources'] == []
                assert body['results'][0]['metrics']['code_test_passed'] is False
            else:
                assert 'retrieval' in body['results'][0]['metrics']
    finally:
        application_app.dependency_overrides.clear()


def test_custom_rubric_scores_without_leaking_reference_into_prompt():
    prompts = []
    class CaptureClient(FakeServiceClient):
        async def generate(self, prompt, model=None, max_tokens=256):
            prompts.append(prompt)
            return await super().generate(prompt, model, max_tokens)
    application_app.dependency_overrides[get_service_client] = CaptureClient
    try:
        response = TestClient(application_app).post('/evaluation/compare-models', json={
            'question': 'Custom deadline question', 'models': ['codellama', 'starcoder2:3b'],
            'reference_answer': 'Assignment 2 is due Friday. PRIVATE_RUBRIC', 'required_facts': ['Friday'],
        })
        assert response.status_code == 200
        metrics = response.json()['results'][0]['metrics']
        assert metrics['accuracy'] == 1
        assert metrics['question_overlap_f1'] >= 0
        assert 'retrieval' not in metrics
        assert all('PRIVATE_RUBRIC' not in p for p in prompts)
    finally:
        application_app.dependency_overrides.clear()


class NoContextClient(FakeServiceClient):
    async def retrieve(self, question: str, top_k: int = 4) -> RetrieveResponse:
        return RetrieveResponse(question=question, grounded_prompt=None, sources=[])

    async def generate(self, prompt: str) -> GenerateResponse:
        raise AssertionError("LLM must not be called without retrieved evidence")


def test_orchestrator_does_not_call_llm_without_context() -> None:
    import asyncio

    orchestrator = Orchestrator(NoContextClient())  # type: ignore[arg-type]
    answer: OrchestratedAnswer = asyncio.run(orchestrator.grounded_answer("Unknown"))
    assert answer.model == "not-called"
    assert answer.sources == []
