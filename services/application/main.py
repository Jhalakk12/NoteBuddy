"""Application Service: public user API and orchestration entry point."""

import asyncio
import os
from time import perf_counter

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from services.application.client import DownstreamServiceError, ServiceClient
from services.application.orchestrator import Orchestrator
from services.contracts import (
    DocumentDeleteResponse,
    EvaluationCompareRequest,
    EvaluationCompareResponse,
    EvaluationModelResult,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    KnowledgeStatus,
    OrchestratedAnswer,
    ServiceStatus,
    ServiceComparison,
    SystemStatus,
    UploadResponse,
    UserQuestion,
)
from week4.api import router as evaluation_router
from week4.evaluator import load_dataset
from week4.live_metrics import live_evidence
from week4.metrics import keyword_accuracy, relevance_f1, hallucination_proxy, retrieval_scores, run_code_test
from services.contracts import RetrieveResponse


from services.application.guardrails_api import router as guardrails_router

app = FastAPI(title="NoteBuddy — Application Service", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://127.0.0.1:8080,http://localhost:8080",
        ).split(",")
        if origin.strip()
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)
app.include_router(evaluation_router)
app.include_router(guardrails_router)


def get_service_client() -> ServiceClient:
    return ServiceClient(
        retrieval_url=os.getenv("RETRIEVAL_SERVICE_URL", "http://127.0.0.1:8001"),
        llm_url=os.getenv("LLM_SERVICE_URL", "http://127.0.0.1:8002"),
        data_url=os.getenv("DATA_SERVICE_URL", "http://127.0.0.1:8003"),
        timeout_seconds=settings.ollama_timeout_seconds + 30,
    )


def get_orchestrator(client: ServiceClient = Depends(get_service_client)) -> Orchestrator:
    return Orchestrator(client, settings.retrieval_top_k)


@app.post("/evaluation/compare-models", response_model=EvaluationCompareResponse)
async def compare_evaluation_models(
    request: EvaluationCompareRequest,
    client: ServiceClient = Depends(get_service_client),
) -> EvaluationCompareResponse:
    """Run one controlled, retrieval-grounded prompt through selected local models."""
    selected_models = list(dict.fromkeys(request.models))
    if len(selected_models) < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Select at least two different models for comparison.",
        )
    approved_models = set(settings.week4_models)
    invalid_models = [model for model in selected_models if model not in approved_models]
    if invalid_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Models are not approved for evaluation: {', '.join(invalid_models)}",
        )

    try:
        if request.domain == 'code':
            retrieved = RetrieveResponse(question=request.question, grounded_prompt=request.question, sources=[])
        elif request.domain == 'repository':
            retrieved = await client.retrieve_repository(request.question, request.top_k)
        else:
            retrieved = await client.retrieve(request.question, request.top_k)
        if request.domain != 'code' and (not retrieved.sources or not retrieved.grounded_prompt):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No evidence was retrieved. Ingest the selected knowledge base before comparing models.",
            )
        results: list[EvaluationModelResult] = []
        rubric = next((item for item in load_dataset() if item['domain'] == request.domain and item['question'].strip().casefold() == request.question.strip().casefold()), None)
        if request.reference_answer.strip():
            rubric = {'reference_answer': request.reference_answer.strip(), 'required_keywords': [fact.strip() for fact in request.required_facts if fact.strip()], 'custom': True}
        for model in selected_models:
            started = perf_counter()
            generated = await client.generate(
                retrieved.grounded_prompt,
                model=model,
                max_tokens=request.max_tokens,
            )
            elapsed = perf_counter() - started
            metrics = live_evidence(request.question, generated.answer, retrieved.sources)
            if rubric:
                context = '\n\n'.join(source.excerpt for source in retrieved.sources) or request.question
                flagged, reasons = hallucination_proxy(generated.answer, context, expect_refusal=rubric.get('expect_refusal', False))
                metrics.update({
                    'grading': 'Automatic rubric proxies, not human grading',
                    'accuracy': keyword_accuracy(generated.answer, rubric.get('required_keywords', []), rubric.get('expect_refusal', False)) if rubric.get('required_keywords') or rubric.get('expect_refusal') else None,
                    'relevance_f1': relevance_f1(generated.answer, rubric['reference_answer']),
                    'hallucination_flag': flagged,
                    'hallucination_reasons': reasons,
                    'reference_answer': rubric['reference_answer'],
                })
                if request.domain != 'code' and not rubric.get('custom'):
                    metrics['retrieval'] = retrieval_scores([s.citation if rubric.get('expected_citations') else s.source for s in retrieved.sources], rubric.get('expected_citations') or rubric.get('expected_sources', []), request.top_k)
                if rubric.get('code_test'):
                    passed, detail = await asyncio.to_thread(run_code_test, generated.answer, rubric['code_test'])
                    metrics.update(code_test_passed=passed, code_test_detail=detail)
            total_tokens = None
            if generated.prompt_tokens is not None or generated.completion_tokens is not None:
                total_tokens = (generated.prompt_tokens or 0) + (generated.completion_tokens or 0)
            results.append(
                EvaluationModelResult(
                    model=generated.model,
                    answer=generated.answer,
                    metrics=metrics,
                    latency_seconds=round(elapsed, 3),
                    prompt_tokens=generated.prompt_tokens,
                    completion_tokens=generated.completion_tokens,
                    total_tokens=total_tokens,
                    model_memory_mb=(
                        round(generated.model_memory_bytes / 1024**2, 2)
                        if generated.model_memory_bytes is not None
                        else None
                    ),
                    gpu_memory_mb=(
                        round(generated.gpu_memory_bytes / 1024**2, 2)
                        if generated.gpu_memory_bytes is not None
                        else None
                    ),
                )
            )
    except DownstreamServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return EvaluationCompareResponse(
        question=request.question,
        sources=retrieved.sources,
        results=results,
        conditions={
            "same_question": True,
            "domain": request.domain,
            "same_retrieved_context": True,
            "temperature": 0.1,
            "max_tokens": request.max_tokens,
            "retrieval_top_k": request.top_k,
        },
        orchestration=["application", "retrieval (once)", *[f"llm:{model}" for model in selected_models]],
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="application")


@app.get("/system/status", response_model=SystemStatus)
async def system_status(
    client: ServiceClient = Depends(get_service_client),
) -> SystemStatus:
    checks = await asyncio.gather(
        client.service_status("retrieval", client.retrieval_url),
        client.service_status("llm + ollama", client.llm_url),
        client.service_status("data", client.data_url),
    )
    services = [
        ServiceStatus(
            name="application",
            url="this service",
            status="ok",
            detail="public API and orchestration",
        ),
        *checks,
    ]
    overall = "ok" if all(item.status == "ok" for item in services) else "degraded"
    return SystemStatus(overall=overall, services=services)


@app.get("/knowledge/status", response_model=KnowledgeStatus)
async def knowledge_status(
    client: ServiceClient = Depends(get_service_client),
) -> KnowledgeStatus:
    try:
        return await client.knowledge_status()
    except DownstreamServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@app.post("/knowledge/upload", response_model=UploadResponse)
async def upload_knowledge(
    files: list[UploadFile] = File(...),
    client: ServiceClient = Depends(get_service_client),
) -> UploadResponse:
    forwarded = [
        (
            upload.filename or "unnamed",
            await upload.read(),
            upload.content_type or "application/octet-stream",
        )
        for upload in files
    ]
    try:
        return await client.upload_documents(forwarded)
    except DownstreamServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@app.delete("/knowledge/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_knowledge_document(
    filename: str,
    client: ServiceClient = Depends(get_service_client),
) -> DocumentDeleteResponse:
    try:
        return await client.delete_document(filename)
    except DownstreamServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@app.post("/knowledge/ingest", response_model=IngestResponse)
async def ingest_knowledge(
    request: IngestRequest,
    client: ServiceClient = Depends(get_service_client),
) -> IngestResponse:
    try:
        return await client.ingest(request.reset)
    except DownstreamServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@app.post("/ask", response_model=OrchestratedAnswer)
async def ask(
    request: UserQuestion,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    guardrails: bool = True,
) -> OrchestratedAnswer:
    try:
        return await orchestrator.grounded_answer(request.question, with_guardrails=guardrails)
    except DownstreamServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@app.post("/compare", response_model=ServiceComparison)
async def compare(
    request: UserQuestion,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> ServiceComparison:
    try:
        raw = await orchestrator.raw_answer(request.question)
        grounded = await orchestrator.grounded_answer(request.question)
    except DownstreamServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return ServiceComparison(
        question=request.question,
        without_rag=raw,
        with_rag=grounded,
    )
