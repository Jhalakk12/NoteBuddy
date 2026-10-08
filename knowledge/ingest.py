"""Exercise 2 ingestion orchestration."""

from dataclasses import dataclass
from pathlib import Path

from knowledge.chunking import chunk_units
from knowledge.loaders import discover_documents, load_document
from knowledge.store import KnowledgeStore


@dataclass(frozen=True)
class IngestionReport:
    documents: int
    units: int
    chunks: int
    collection_total: int


def ingest_directory(
    document_directory: Path,
    store: KnowledgeStore,
    *,
    reset: bool = False,
    max_characters: int = 1_200,
    overlap_characters: int = 200,
) -> IngestionReport:
    documents = discover_documents(document_directory)
    if not documents:
        raise FileNotFoundError(
            f"No supported PDF, TXT, or Markdown documents found in {document_directory}"
        )
    if reset:
        store.reset()

    all_units = []
    for path in documents:
        all_units.extend(load_document(path, document_directory))
    chunks = chunk_units(all_units, max_characters, overlap_characters)
    store.upsert(chunks)
    return IngestionReport(
        documents=len(documents),
        units=len(all_units),
        chunks=len(chunks),
        collection_total=store.count(),
    )

