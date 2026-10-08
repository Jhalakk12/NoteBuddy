"""Exercise 2 loader, chunking, metadata, and Chroma tests."""

from pathlib import Path
import asyncio

import pymupdf

from knowledge.chunking import chunk_units
from knowledge.ingest import ingest_directory
from knowledge.loaders import load_document
from knowledge.models import RetrievedChunk
from knowledge.store import KnowledgeStore
from knowledge.rag import (
    NO_CONTEXT_ANSWER,
    RagService,
    build_grounded_prompt,
    ensure_citation,
)


class FakeEmbedder:
    model = "fake-embedding-model"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.5] for text in texts]


def test_markdown_headings_become_citation_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "syllabus.md"
    path.write_text(
        "# Assessments\nAssignment 2 is due Friday.\n\n# Midterm\nWeeks 1-5.",
        encoding="utf-8",
    )
    units = load_document(path, tmp_path)
    chunks = chunk_units(units, max_characters=200, overlap_characters=20)

    assert [chunk.section for chunk in chunks] == ["Assessments", "Midterm"]
    assert chunks[0].metadata() == {
        "source": "syllabus.md",
        "page": 0,
        "section": "Assessments",
        "chunk_index": 1,
        "citation": "syllabus.md, section: Assessments",
    }


def test_pdf_pages_are_preserved_in_metadata(tmp_path: Path) -> None:
    path = tmp_path / "lecture-04.pdf"
    document = pymupdf.open()
    for text in ("Slide one: Recursion", "Slide two: Base cases"):
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()

    units = load_document(path, tmp_path)
    chunks = chunk_units(units, max_characters=200, overlap_characters=20)

    assert [chunk.page for chunk in chunks] == [1, 2]
    assert chunks[1].citation == "lecture-04.pdf, page/slide 2"


def test_ingestion_persists_chunks_and_metadata(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "assignment.txt").write_text(
        "Assignment 2 is due on 15 October.", encoding="utf-8"
    )
    store = KnowledgeStore(tmp_path / "chroma", FakeEmbedder())

    report = ingest_directory(documents, store, reset=True)
    sample = store.sample(limit=1)

    assert report.documents == 1
    assert report.chunks == 1
    assert report.collection_total == 1
    assert sample["metadatas"][0]["source"] == "assignment.txt"
    assert sample["metadatas"][0]["citation"] == (
        "assignment.txt, section: Document"
    )


def test_similarity_query_returns_citation_metadata(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "syllabus.md").write_text(
        "# Assessment\nAssignment 2 is due Friday.", encoding="utf-8"
    )
    store = KnowledgeStore(tmp_path / "chroma", FakeEmbedder())
    ingest_directory(documents, store, reset=True)

    result = store.query("When is assignment 2 due?", top_k=1)

    assert len(result) == 1
    assert result[0].citation == "syllabus.md, section: Assessment"
    assert "due Friday" in result[0].text


def test_grounded_prompt_includes_rules_context_and_exact_citation() -> None:
    source = RetrievedChunk(
        text="The midterm covers Weeks 1-5.",
        source="syllabus.md",
        page=2,
        section="Page 2",
        citation="syllabus.md, page/slide 2",
        distance=0.2,
    )
    prompt = build_grounded_prompt("What does the midterm cover?", [source])
    assert "using ONLY the supplied course context" in prompt
    assert "[Source 1: syllabus.md, page/slide 2]" in prompt
    assert source.text in prompt


def test_citation_safeguard_appends_highest_ranked_source() -> None:
    source = RetrievedChunk(
        text="Assignment 2 is due Friday.",
        source="syllabus.md",
        page=None,
        section="Assessment",
        citation="syllabus.md, section: Assessment",
        distance=0.1,
    )
    answer = ensure_citation("Assignment 2 is due Friday.", [source])
    assert answer.endswith("[syllabus.md, section: Assessment]")


class EmptyStore:
    def query(self, question: str, top_k: int) -> list:
        return []


class RecordingLlm:
    model = "test"

    async def generate(self, prompt: str) -> str:
        raise AssertionError("LLM should not be called without retrieved context")


def test_rag_declines_without_course_context() -> None:
    service = RagService(EmptyStore(), RecordingLlm())  # type: ignore[arg-type]
    answer, sources = asyncio.run(service.answer("Unknown question"))
    assert answer == NO_CONTEXT_ANSWER
    assert sources == []
