"""Data models and contracts for NoteBuddy Guardrails and AI Output Testing."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class GuardrailAction(str, Enum):
    ALLOWED = "allowed"
    REJECTED = "rejected"
    INTERCEPTED_REFUSAL = "intercepted_refusal"
    SANITIZED = "sanitized"


class GuardrailStage(str, Enum):
    INPUT = "input"
    RETRIEVAL = "retrieval"
    OUTPUT = "output"


class GuardrailCheckResult(BaseModel):
    name: str
    stage: GuardrailStage
    passed: bool
    action: GuardrailAction
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0


class GuardrailEvaluation(BaseModel):
    passed: bool
    action: GuardrailAction
    primary_reason: str | None = None
    checks: list[GuardrailCheckResult] = Field(default_factory=list)
    sanitized_output: str | None = None
    total_guardrail_latency_ms: float = 0.0


class OutputTestCondition(str, Enum):
    RELEVANCE = "relevance"
    CONTEXT_GROUNDEDNESS = "context_groundedness"
    NO_UNSUPPORTED_CLAIMS = "no_unsupported_claims"
    FORMAT_COMPLIANCE = "format_compliance"
    ANSWERS_WHEN_SUFFICIENT = "answers_when_sufficient"
    REFUSES_WHEN_UNAVAILABLE = "refuses_when_unavailable"


class SingleConditionResult(BaseModel):
    condition: OutputTestCondition
    passed: bool
    score: float
    threshold: float
    description: str
    evidence: str


class OutputTestCaseResult(BaseModel):
    test_id: str
    category: str
    question: str
    context_available: bool
    answer: str
    all_passed: bool
    conditions: list[SingleConditionResult]
    guardrail_status: GuardrailEvaluation | None = None


class OutputTestSuiteSummary(BaseModel):
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float
    condition_pass_rates: dict[str, float]
    tests: list[OutputTestCaseResult]
    tested_at: str
