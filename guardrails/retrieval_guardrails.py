"""Retrieval guardrails: verifies sufficient information availability and confidence before LLM generation."""

from __future__ import annotations

import time
from guardrails.models import GuardrailAction, GuardrailCheckResult, GuardrailStage
from services.contracts import SourceDto

MIN_CONFIDENCE_THRESHOLD = 0.25


def check_retrieval_confidence(
    sources: list[SourceDto],
    threshold: float = MIN_CONFIDENCE_THRESHOLD,
) -> GuardrailCheckResult:
    """Verifies that retrieved chunks have sufficient relevance to prevent hallucination."""
    start = time.perf_counter()
    latency_ms = (time.perf_counter() - start) * 1000

    if not sources:
        return GuardrailCheckResult(
            name="retrieval_confidence",
            stage=GuardrailStage.RETRIEVAL,
            passed=False,
            action=GuardrailAction.INTERCEPTED_REFUSAL,
            reason="No relevant course or repository documents were found for this query.",
            details={"sources_count": 0, "threshold": threshold},
            latency_ms=latency_ms,
        )

    best_score = max(source.relevance_score for source in sources)
    if best_score < threshold:
        return GuardrailCheckResult(
            name="retrieval_confidence",
            stage=GuardrailStage.RETRIEVAL,
            passed=False,
            action=GuardrailAction.INTERCEPTED_REFUSAL,
            reason=f"Insufficient information available: maximum retrieval confidence ({best_score:.2f}) is below the required reliability threshold ({threshold:.2f}).",
            details={
                "best_score": round(best_score, 3),
                "threshold": threshold,
                "sources_count": len(sources),
            },
            latency_ms=latency_ms,
        )

    return GuardrailCheckResult(
        name="retrieval_confidence",
        stage=GuardrailStage.RETRIEVAL,
        passed=True,
        action=GuardrailAction.ALLOWED,
        reason=f"Sufficient evidence retrieved with confidence {best_score:.2f} >= {threshold:.2f}.",
        details={"best_score": round(best_score, 3), "sources_count": len(sources)},
        latency_ms=latency_ms,
    )
