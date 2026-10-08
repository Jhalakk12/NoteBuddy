"""Shared HTTP contracts for the Exercise 4 services."""

from pydantic import BaseModel, Field
from typing import Literal


class HealthResponse(BaseModel):
    service: str
    status: str = "ok"
    details: dict[str, str | int] = Field(default_factory=dict)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=100_000)
    model: str | None = Field(default=None, min_length=1, max_length=100)
    max_tokens: int = Field(default=256, ge=16, le=1024)


class GenerateResponse(BaseModel):
    answer: str
    model: str
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    eval_duration_ns: int | None = None
    model_memory_bytes: int | None = None
    gpu_memory_bytes: int | None = None


class ModelOption(BaseModel):
    id: str
    label: str
    available: bool


class ModelListResponse(BaseModel):
    default_model: str
    models: list[ModelOption]


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=4, ge=1, le=20)


class SourceDto(BaseModel):
    citation: str
    source: str
    page: int | None
    section: str
    excerpt: str
    relevance_score: float


class EvaluationCompareRequest(BaseModel):
    reference_answer: str = Field(default='', max_length=10000)
    required_facts: list[str] = Field(default_factory=list, max_length=30)
    domain: Literal['course', 'repository', 'code'] = 'course'
    question: str = Field(min_length=1, max_length=10_000)
    models: list[str] = Field(min_length=2, max_length=3)
    top_k: int = Field(default=4, ge=1, le=8)
    max_tokens: int = Field(default=160, ge=16, le=512)


class EvaluationModelResult(BaseModel):
    metrics: dict = Field(default_factory=dict)
    model: str
    answer: str
    latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    model_memory_mb: float | None = None
    gpu_memory_mb: float | None = None


class EvaluationCompareResponse(BaseModel):
    question: str
    sources: list[SourceDto]
    results: list[EvaluationModelResult]
    conditions: dict[str, str | int | float | bool]
    orchestration: list[str]


class RetrieveResponse(BaseModel):
    question: str
    grounded_prompt: str | None
    sources: list[SourceDto]


class IngestRequest(BaseModel):
    reset: bool = False


class IngestResponse(BaseModel):
    documents: int
    units: int
    chunks: int
    collection_total: int
    embedding_model: str


class RepositoryIngestResponse(BaseModel):
    files: int
    chunks: int
    collection_total: int
    embedding_model: str


class DocumentInfo(BaseModel):
    name: str
    extension: str
    size_bytes: int


class KnowledgeStatus(BaseModel):
    documents: list[DocumentInfo]
    document_count: int
    collection_total: int
    supported_extensions: list[str]


class UploadResponse(BaseModel):
    uploaded: list[DocumentInfo]
    rejected: list[str]


class DocumentDeleteResponse(BaseModel):
    deleted: DocumentInfo
    remaining_documents: int
    collection_total: int


class ServiceStatus(BaseModel):
    name: str
    url: str
    status: str
    detail: str | None = None


class SystemStatus(BaseModel):
    overall: str
    services: list[ServiceStatus]


class UserQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)


class OrchestratedAnswer(BaseModel):
    answer: str
    sources: list[SourceDto]
    model: str
    orchestration: list[str]
    exercise: int = 4
    guardrail: dict | None = None


class RawAnswer(BaseModel):
    answer: str
    model: str


class ServiceComparison(BaseModel):
    question: str
    without_rag: RawAnswer
    with_rag: OrchestratedAnswer


class GuardrailCompareResult(BaseModel):
    mode: str
    answer: str
    sources: list[SourceDto] = Field(default_factory=list)
    guardrail_action: str
    guardrail_reason: str | None = None
    latency_seconds: float
    conditions_passed: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    problematic: bool = False
    problem_description: str | None = None


class GuardrailCompareResponse(BaseModel):
    question: str
    without_guardrail: GuardrailCompareResult
    with_guardrail: GuardrailCompareResult
    effectiveness_summary: str
