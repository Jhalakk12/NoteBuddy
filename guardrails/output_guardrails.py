"""Output guardrails: inspects, verifies, and sanitizes LLM output before presentation to user."""

from __future__ import annotations

import re
import time
from guardrails.models import GuardrailAction, GuardrailCheckResult, GuardrailStage
from knowledge.rag import NO_CONTEXT_ANSWER
from services.contracts import SourceDto

PROMPT_LEAK_PATTERNS = [
    re.compile(r"^(?:system|human|assistant|user|context):\s*", re.I | re.M),
    re.compile(r"(?:<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>)", re.I),
]

NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
CITATION_RE = re.compile(r"\[([^\]]+)\]")
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*", re.I)
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on", "for",
    "and", "or", "with", "what", "when", "where", "how", "does", "do", "it", "this", "that",
    "not", "by", "as", "at", "from", "which", "will", "can", "has", "have", "had", "we", "you",
}


def sanitize_prompt_leakage(answer: str) -> tuple[str, bool]:
    """Removes internal model artifacts or prompt wrappers."""
    cleaned = answer
    modified = False
    for pattern in PROMPT_LEAK_PATTERNS:
        if pattern.search(cleaned):
            cleaned = pattern.sub("", cleaned)
            modified = True
    return cleaned.strip(), modified


def check_output_format(answer: str, is_refusal: bool) -> GuardrailCheckResult:
    """Checks that the generated output follows the expected academic response format."""
    start = time.perf_counter()
    latency_ms = (time.perf_counter() - start) * 1000

    cleaned, leaked = sanitize_prompt_leakage(answer)
    if leaked:
        return GuardrailCheckResult(
            name="format_compliance",
            stage=GuardrailStage.OUTPUT,
            passed=False,
            action=GuardrailAction.SANITIZED,
            reason="Prompt or system delimiter leakage detected in LLM response; sanitized.",
            details={"sanitized_sample": cleaned[:100]},
            latency_ms=latency_ms,
        )

    if not is_refusal and not CITATION_RE.search(cleaned):
        return GuardrailCheckResult(
            name="format_compliance",
            stage=GuardrailStage.OUTPUT,
            passed=False,
            action=GuardrailAction.SANITIZED,
            reason="Non-refusal answer lacked mandatory citation brackets; will be attached by orchestrator.",
            details={},
            latency_ms=latency_ms,
        )

    return GuardrailCheckResult(
        name="format_compliance",
        stage=GuardrailStage.OUTPUT,
        passed=True,
        action=GuardrailAction.ALLOWED,
        reason="Output satisfies expected format standards.",
        details={},
        latency_ms=latency_ms,
    )


def check_unsupported_claims(
    answer: str,
    context: str,
    is_refusal: bool,
) -> GuardrailCheckResult:
    """Detects numeric facts, percentages, or dates present in the answer but absent from context."""
    start = time.perf_counter()
    latency_ms = (time.perf_counter() - start) * 1000

    if is_refusal or not context:
        return GuardrailCheckResult(
            name="unsupported_claims",
            stage=GuardrailStage.OUTPUT,
            passed=True,
            action=GuardrailAction.ALLOWED,
            reason="Refusal or empty context does not produce factual claims.",
            details={},
            latency_ms=latency_ms,
        )

    answer_numbers = set(NUMBER_RE.findall(answer))
    context_numbers = set(NUMBER_RE.findall(context))
    unsupported_numbers = answer_numbers - context_numbers

    # Filter out harmless trivial numbers like 1, 2 for bullet lists
    meaningful_unsupported = {n for n in unsupported_numbers if not (n.isdigit() and int(n) in (1, 2, 3))}

    if meaningful_unsupported:
        return GuardrailCheckResult(
            name="unsupported_claims",
            stage=GuardrailStage.OUTPUT,
            passed=False,
            action=GuardrailAction.INTERCEPTED_REFUSAL,
            reason=f"Hallucination detected: answer contains numeric claims not supported by retrieved context: {sorted(meaningful_unsupported)}.",
            details={"unsupported_numbers": sorted(meaningful_unsupported)},
            latency_ms=latency_ms,
        )

    # Token-level groundedness ratio
    answer_tokens = set(TOKEN_RE.findall(answer.casefold())) - STOP_WORDS
    context_tokens = set(TOKEN_RE.findall(context.casefold())) - STOP_WORDS
    if answer_tokens:
        grounded_ratio = len(answer_tokens & context_tokens) / len(answer_tokens)
    else:
        grounded_ratio = 1.0

    if grounded_ratio < 0.20:
        return GuardrailCheckResult(
            name="unsupported_claims",
            stage=GuardrailStage.OUTPUT,
            passed=False,
            action=GuardrailAction.INTERCEPTED_REFUSAL,
            reason=f"Insufficient context grounding: only {grounded_ratio * 100:.1f}% of answer content is supported by retrieved context.",
            details={"grounded_ratio": round(grounded_ratio, 3)},
            latency_ms=latency_ms,
        )

    return GuardrailCheckResult(
        name="unsupported_claims",
        stage=GuardrailStage.OUTPUT,
        passed=True,
        action=GuardrailAction.ALLOWED,
        reason=f"All numeric claims are verified against context ({len(answer_numbers & context_numbers)} numbers matched, {grounded_ratio * 100:.1f}% token overlap).",
        details={"grounded_ratio": round(grounded_ratio, 3)},
        latency_ms=latency_ms,
    )
