"""Retrieval/RAG Service: vector search and grounded-prompt construction."""

from fastapi import Depends, FastAPI, HTTPException, status

from app.config import settings
from knowledge.embeddings import EmbeddingError, OllamaEmbeddingClient
from knowledge.rag import build_grounded_prompt
from knowledge.store import KnowledgeStore
from services.contracts import (
    HealthResponse,
    RetrieveRequest,
    RetrieveResponse,
    SourceDto,
)


app = FastAPI(title="NoteBuddy — Retrieval/RAG Service", version="0.4.0")


def get_store() -> KnowledgeStore:
    embedder = OllamaEmbeddingClient(
        settings.ollama_base_url,
        settings.embedding_model,
        settings.ollama_timeout_seconds,
    )
    return KnowledgeStore(settings.chroma_path, embedder)


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
def health(store: KnowledgeStore = Depends(get_store)) -> HealthResponse:
    return HealthResponse(
        service="retrieval",
        details={
            "chunks": store.count(),
            "embedding_model": settings.embedding_model,
        },
    )


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    request: RetrieveRequest,
    store: KnowledgeStore = Depends(get_store),
) -> RetrieveResponse:
    try:
        chunks = store.query(request.question, request.top_k)
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    sources = [
        SourceDto(
            citation=chunk.citation,
            source=chunk.source,
            page=chunk.page,
            section=chunk.section,
            excerpt=chunk.text,
            relevance_score=round(chunk.relevance_score, 4),
        )
        for chunk in chunks
    ]
    prompt = build_grounded_prompt(request.question, chunks) if chunks else None
    return RetrieveResponse(
        question=request.question,
        grounded_prompt=prompt,
        sources=sources,
    )


@app.post("/retrieve/repository", response_model=RetrieveResponse)
def retrieve_repository(
    request: RetrieveRequest,
    store: KnowledgeStore = Depends(get_repository_store),
) -> RetrieveResponse:
    try:
        chunks = store.query(request.question, request.top_k)
    except EmbeddingError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    sources = [
        SourceDto(
            citation=chunk.citation,
            source=chunk.source,
            page=chunk.page,
            section=chunk.section,
            excerpt=chunk.text,
            relevance_score=round(chunk.relevance_score, 4),
        )
        for chunk in chunks
    ]
    prompt = _repository_prompt(request.question, sources) if sources else None
    return RetrieveResponse(question=request.question, grounded_prompt=prompt, sources=sources)


@app.get("/sourcegraph/status")
def sourcegraph_status() -> dict:
    from services.retrieval.sourcegraph import get_sourcegraph_engine
    engine = get_sourcegraph_engine()
    return {
        "status": "connected",
        "engine": "Sourcegraph OSS / AST Code Intelligence",
        "indexed_symbols": len(engine.symbols),
        "indexed_files": len(engine.file_index),
    }


@app.get("/sourcegraph/search")
def sourcegraph_search(q: str = "orchestrator", limit: int = 10) -> dict:
    from services.retrieval.sourcegraph import get_sourcegraph_engine
    engine = get_sourcegraph_engine()
    return engine.search(q, limit=limit)


def _repository_prompt(question: str, sources: list[SourceDto]) -> str:
    from services.retrieval.sourcegraph import get_sourcegraph_engine
    engine = get_sourcegraph_engine()
    symbol_context = engine.get_symbol_context(question)
    context = "\n\n".join(
        f"[Source {index}: {source.citation}]\n{source.excerpt}"
        for index, source in enumerate(sources, start=1)
    )
    if symbol_context:
        context = f"{symbol_context}\n\n{context}"
    return f"""You are analysing the NoteBuddy repository with Sourcegraph code intelligence. Answer using the retrieved code and symbol context.
Explain the relationship across files, cite every claim with the exact source label, and say when the retrieved files are insufficient.

REPOSITORY CONTEXT
{context}

QUESTION
{question}

ANSWER
"""
