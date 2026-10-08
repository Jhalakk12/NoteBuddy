"""Week 4 dataset, metrics, repository indexing, and dashboard API tests."""

from pathlib import Path

from fastapi.testclient import TestClient

from knowledge.store import KnowledgeStore
from services.application.main import app as application_app
from week4.evaluator import load_dataset
from week4.metrics import (
    hallucination_proxy,
    keyword_accuracy,
    retrieval_scores,
    run_code_test,
)
from week4.repository import ingest_repository


class FakeEmbedder:
    model = "fake"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


def test_shared_evaluation_dataset_has_required_scope() -> None:
    dataset = load_dataset()
    assert 20 <= len(dataset) <= 30
    assert len({item["id"] for item in dataset}) == len(dataset)
    assert {item["domain"] for item in dataset} == {"course", "repository", "code"}
    assert sum(bool(item.get("code_test")) for item in dataset) >= 3


def test_quantitative_metric_formulas() -> None:
    assert keyword_accuracy("Due at 5 PM on Friday", ["5 PM", "Friday"]) == 1.0
    scores = retrieval_scores(["wrong.md", "syllabus.md"], ["syllabus.md"], 4)
    assert scores == {"precision_at_k": 0.25, "recall_at_k": 1.0, "mrr": 0.5}
    hallucinated, reasons = hallucination_proxy("It is due on 31 October.", "Due on 15 October.")
    assert hallucinated
    assert "unsupported_number:31" in reasons


def test_generated_code_runs_fixed_unit_tests() -> None:
    passed, detail = run_code_test(
        "def normalize_question(text):\n    return ' '.join(text.split())",
        "normalize_question",
    )
    assert passed, detail


def test_repository_index_preserves_file_and_line_citations(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "module.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
    store = KnowledgeStore(tmp_path / "chroma", FakeEmbedder())
    report = ingest_repository(root, store)
    sample = store.sample(1)
    assert report.files == 1
    assert sample["metadatas"][0]["source"] == "module.py"
    assert "lines 1-2" in sample["metadatas"][0]["citation"]


def test_week4_dashboard_exposes_same_application_dataset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EVALUATION_PATH", str(tmp_path))
    client = TestClient(application_app)
    overview = client.get("/evaluation/overview")
    dataset = client.get("/evaluation/dataset")
    assert overview.status_code == 200
    assert overview.json()["same_application"] is True
    assert overview.json()["status"]["state"] == "not_run"
    assert dataset.json()["count"] == 25
