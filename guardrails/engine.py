"""Unified Guardrails Engine coordinating input, retrieval, and output controls."""

from __future__ import annotations

import time
from guardrails.input_guardrails import check_input_length, check_prompt_injection, check_scope
from guardrails.models import GuardrailAction, GuardrailCheckResult, GuardrailEvaluation
from guardrails.output_guardrails import check_output_format, check_unsupported_claims, sanitize_prompt_leakage
from guardrails.retrieval_guardrails import check_retrieval_confidence
from knowledge.rag import NO_CONTEXT_ANSWER
from services.contracts import SourceDto


class GuardrailEngine:
    """Orchestrates comprehensive safety and quality checks across the RAG lifecycle."""

    @staticmethod
    def evaluate_input(question: str) -> GuardrailEvaluation:
        start = time.perf_counter()
        checks: list[GuardrailCheckResult] = [
            check_input_length(question),
            check_prompt_injection(question),
            check_scope(question),
        ]

        failed_check = next((c for c in checks if not c.passed), None)
        latency_ms = (time.perf_counter() - start) * 1000

        if failed_check:
            return GuardrailEvaluation(
                passed=False,
                action=failed_check.action,
                primary_reason=failed_check.reason,
                checks=checks,
                sanitized_output=NO_CONTEXT_ANSWER if failed_check.action == GuardrailAction.INTERCEPTED_REFUSAL else f"Request rejected by guardrail: {failed_check.reason}",
                total_guardrail_latency_ms=latency_ms,
            )

        return GuardrailEvaluation(
            passed=True,
            action=GuardrailAction.ALLOWED,
            primary_reason=None,
            checks=checks,
            total_guardrail_latency_ms=latency_ms,
        )

    @staticmethod
    def evaluate_retrieval(sources: list[SourceDto], threshold: float = 0.25) -> GuardrailEvaluation:
        start = time.perf_counter()
        check = check_retrieval_confidence(sources, threshold)
        latency_ms = (time.perf_counter() - start) * 1000

        if not check.passed:
            return GuardrailEvaluation(
                passed=False,
                action=check.action,
                primary_reason=check.reason,
                checks=[check],
                sanitized_output=NO_CONTEXT_ANSWER,
                total_guardrail_latency_ms=latency_ms,
            )

        return GuardrailEvaluation(
            passed=True,
            action=GuardrailAction.ALLOWED,
            primary_reason=None,
            checks=[check],
            total_guardrail_latency_ms=latency_ms,
        )

    @staticmethod
    def evaluate_output(
        answer: str,
        context: str,
        is_refusal: bool,
    ) -> GuardrailEvaluation:
        start = time.perf_counter()
        sanitized_text, _ = sanitize_prompt_leakage(answer)

        checks: list[GuardrailCheckResult] = [
            check_output_format(sanitized_text, is_refusal),
            check_unsupported_claims(sanitized_text, context, is_refusal),
        ]

        failed_check = next((c for c in checks if not c.passed and c.action != GuardrailAction.SANITIZED), None)
        sanitized_check = next((c for c in checks if c.action == GuardrailAction.SANITIZED), None)
        latency_ms = (time.perf_counter() - start) * 1000

        if failed_check:
            return GuardrailEvaluation(
                passed=False,
                action=failed_check.action,
                primary_reason=failed_check.reason,
                checks=checks,
                sanitized_output=NO_CONTEXT_ANSWER,
                total_guardrail_latency_ms=latency_ms,
            )

        return GuardrailEvaluation(
            passed=True,
            action=GuardrailAction.SANITIZED if sanitized_check else GuardrailAction.ALLOWED,
            primary_reason=sanitized_check.reason if sanitized_check else None,
            checks=checks,
            sanitized_output=sanitized_text,
            total_guardrail_latency_ms=latency_ms,
        )
