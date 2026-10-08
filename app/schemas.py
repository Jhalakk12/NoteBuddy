"""API request and response contracts for Exercise 1."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)


class AskResponse(BaseModel):
    answer: str
    model: str
    exercise: int = 1


class SourceReference(BaseModel):
    citation: str
    source: str
    page: int | None
    section: str
    excerpt: str
    relevance_score: float


class RagResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    model: str
    exercise: int = 3


class CompareResponse(BaseModel):
    question: str
    without_rag: AskResponse
    with_rag: RagResponse


class HealthResponse(BaseModel):
    status: str
    ollama_url: str
    model: str
