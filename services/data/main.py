"""Data Service: controlled document ingestion and knowledge-base management."""

import os
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status

from app.config import settings
from knowledge.embeddings import EmbeddingError, OllamaEmbeddingClient
from knowledge.ingest import ingest_directory
from knowledge.loaders import SUPPORTED_EXTENSIONS, discover_documents
from knowledge.store import KnowledgeStore
from week4.repository import discover_repository_files, ingest_repository
from services.contracts import (
    DocumentDeleteResponse,
    DocumentInfo,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    KnowledgeStatus,
    RepositoryIngestResponse,
    UploadResponse,
)


app = FastAPI(title="NoteBuddy — Data Service", version="0.4.0")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def get_document_path() -> Path:
    return Path(os.getenv("DOCUMENT_PATH", "data/documents"))


def get_store() -> KnowledgeStore:
    embedder = OllamaEmbeddingClient(
        settings.ollama_base_url,
        settings.embedding_model,
        settings.ollama_timeout_seconds,
    )
    return KnowledgeStore(settings.chroma_path, embedder)


def get_repository_path() -> Path:
    return Path(os.getenv("REPOSITORY_PATH", Path(__file__).resolve().parents[2]))


def get_repository_store() -> KnowledgeStore:
    embedder = OllamaEmbeddingClient(
        settings.ollama_base_url,
        settings.embedding_model,
        settings.ollama_timeout_seconds,
    )
    return KnowledgeStore(
        settings.chroma_path,
        embedder,
        collection_name="notebuddy_repository",
    )


@app.get("/health", response_model=HealthResponse)
def health(
    store: KnowledgeStore = Depends(get_store),
    document_path: Path = Depends(get_document_path),
) -> HealthResponse:
    document_path.mkdir(parents=True, exist_ok=True)
    return HealthResponse(
        service="data",
        details={
            "documents": len(discover_documents(document_path)),
            "chunks": store.count(),
        },
    )


@app.get("/documents", response_model=KnowledgeStatus)
def documents(
    store: KnowledgeStore = Depends(get_store),
    document_path: Path = Depends(get_document_path),
) -> KnowledgeStatus:
    document_path.mkdir(parents=True, exist_ok=True)
    files = [_document_info(path, document_path) for path in discover_documents(document_path)]
    return KnowledgeStatus(
        documents=files,
        document_count=len(files),
        collection_total=store.count(),
        supported_extensions=sorted(SUPPORTED_EXTENSIONS),
    )


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    document_path: Path = Depends(get_document_path),
) -> UploadResponse:
    document_path.mkdir(parents=True, exist_ok=True)
    uploaded: list[DocumentInfo] = []
    rejected: list[str] = []
    for upload in files:
        original_name = upload.filename or "unnamed"
        safe_name = Path(original_name).name
        extension = Path(safe_name).suffix.lower()
        if not safe_name or safe_name != original_name or extension not in SUPPORTED_EXTENSIONS:
            rejected.append(f"{original_name}: unsupported or unsafe filename")
            continue
        content = await upload.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            rejected.append(f"{original_name}: exceeds the 25 MB limit")
            continue
        destination = document_path / safe_name
        destination.write_bytes(content)
        uploaded.append(_document_info(destination, document_path))
    return UploadResponse(uploaded=uploaded, rejected=rejected)


@app.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
def delete_document(
    filename: str,
    store: KnowledgeStore = Depends(get_store),
    document_path: Path = Depends(get_document_path),
) -> DocumentDeleteResponse:
    """Remove a document's vectors and file without rebuilding unrelated sources."""
    safe_name = Path(filename).name
    if not safe_name or safe_name != filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsafe document filename",
        )
    target = document_path / safe_name
    if target.suffix.lower() not in SUPPORTED_EXTENSIONS or not target.is_file() or target.is_symlink():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {safe_name}",
        )

    deleted = _document_info(target, document_path)
    store.delete_source(safe_name)
    target.unlink()
    remaining = discover_documents(document_path)
    collection_total = store.count()
    return DocumentDeleteResponse(
        deleted=deleted,
        remaining_documents=len(remaining),
        collection_total=collection_total,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
    store: KnowledgeStore = Depends(get_store),
    document_path: Path = Depends(get_document_path),
) -> IngestResponse:
    try:
        report = ingest_directory(document_path, store, reset=request.reset)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return IngestResponse(
        **report.__dict__,
        embedding_model=store.embedder.model,
    )


@app.get("/repository/status", response_model=RepositoryIngestResponse)
def repository_status(
    store: KnowledgeStore = Depends(get_repository_store),
    repository_path: Path = Depends(get_repository_path),
) -> RepositoryIngestResponse:
    return RepositoryIngestResponse(
        files=len(discover_repository_files(repository_path)),
        chunks=store.count(),
        collection_total=store.count(),
        embedding_model=store.embedder.model,
    )


@app.post("/repository/ingest", response_model=RepositoryIngestResponse)
def repository_ingest(
    request: IngestRequest,
    store: KnowledgeStore = Depends(get_repository_store),
    repository_path: Path = Depends(get_repository_path),
) -> RepositoryIngestResponse:
    try:
        report = ingest_repository(repository_path, store, reset=request.reset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except EmbeddingError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return RepositoryIngestResponse(**report.__dict__, embedding_model=store.embedder.model)


def _document_info(path: Path, root: Path) -> DocumentInfo:
    return DocumentInfo(
        name=path.relative_to(root).as_posix(),
        extension=path.suffix.lower(),
        size_bytes=path.stat().st_size,
    )
