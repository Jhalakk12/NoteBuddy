"""Exercise 1: minimal User -> FastAPI -> Ollama -> Code Llama flow."""

from fastapi import Depends, FastAPI, HTTPException, status

from app.config import settings
from app.ollama_client import OllamaClient, OllamaError
from app.schemas import (
    AskRequest,
    AskResponse,
    CompareResponse,
    HealthResponse,
    RagResponse,
    SourceReference,
)
from knowledge.embeddings import OllamaEmbeddingClient
from knowledge.rag import RagService
from knowledge.store import KnowledgeStore


app = FastAPI(
    title="NoteBuddy",
    description="Exercise 1: raw Code Llama responses through Ollama.",
    version="0.1.0",
)


def get_ollama_client() -> OllamaClient:
    return OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
    )


def get_rag_service() -> RagService:
    llm = get_ollama_client()
    embedder = OllamaEmbeddingClient(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    store = KnowledgeStore(settings.chroma_path, embedder)
    return RagService(store=store, llm=llm, top_k=settings.retrieval_top_k)


def _rag_response(
    answer: str,
    sources: list,
    model: str,
) -> RagResponse:
    return RagResponse(
        answer=answer,
        model=model,
        sources=[
            SourceReference(
                citation=source.citation,
                source=source.source,
                page=source.page,
                section=source.section,
                excerpt=source.text,
                relevance_score=round(source.relevance_score, 4),
            )
            for source in sources
        ],
    )


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    return {
        "application": "NoteBuddy",
        "exercise": "1 - Basic LLM Application",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        ollama_url=settings.ollama_base_url,
        model=settings.ollama_model,
    )


@app.post("/ask", response_model=AskResponse, tags=["llm"])
async def ask(
    request: AskRequest,
    client: OllamaClient = Depends(get_ollama_client),
) -> AskResponse:
    try:
        answer = await client.generate(request.question)
    except OllamaError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return AskResponse(answer=answer, model=client.model)


@app.post("/ask/rag", response_model=RagResponse, tags=["rag"])
async def ask_with_rag(
    request: AskRequest,
    rag: RagService = Depends(get_rag_service),
) -> RagResponse:
    try:
        answer, sources = await rag.answer(request.question)
    except OllamaError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return _rag_response(answer, sources, rag.llm.model)


@app.post("/compare", response_model=CompareResponse, tags=["rag"])
async def compare_rag(
    request: AskRequest,
    client: OllamaClient = Depends(get_ollama_client),
    rag: RagService = Depends(get_rag_service),
) -> CompareResponse:
    try:
        raw_answer = await client.generate(request.question)
        grounded_answer, sources = await rag.answer(request.question)
    except OllamaError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return CompareResponse(
        question=request.question,
        without_rag=AskResponse(answer=raw_answer, model=client.model),
        with_rag=_rag_response(grounded_answer, sources, rag.llm.model),
    )
