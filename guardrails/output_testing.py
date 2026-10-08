"""AI Output Testing Suite: Evaluates model responses against formal quality, safety, and grounding conditions."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from guardrails.models import (
    GuardrailEvaluation,
    OutputTestCaseResult,
    OutputTestCondition,
    OutputTestSuiteSummary,
    SingleConditionResult,
)
from knowledge.rag import NO_CONTEXT_ANSWER
from services.contracts import SourceDto

NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
CITATION_RE = re.compile(r"\[([^\]]+)\]")
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*", re.I)
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on", "for",
    "and", "or", "with", "what", "when", "where", "how", "does", "do", "it", "this", "that",
    "not", "by", "as", "at", "from", "which", "will", "can", "has", "have", "had", "we", "you",
}

TEST_CASES_PATH = Path(__file__).with_name("test_cases.json")


def load_test_cases() -> list[dict[str, Any]]:
    return json.loads(TEST_CASES_PATH.read_text(encoding="utf-8"))


def is_refusal_answer(answer: str) -> bool:
    lowered = answer.lower()
    return (
        NO_CONTEXT_ANSWER.lower() in lowered
        or "cannot find" in lowered
        or "could not find" in lowered
        or "not provided" in lowered
        or "do not contain" in lowered
        or "does not contain" in lowered
        or "insufficient" in lowered
        or "rejected" in lowered
        or "outside" in lowered
        or "prohibited" in lowered
    )


def evaluate_output_conditions(
    question: str,
    answer: str,
    context: str,
    context_available: bool,
    guardrail_status: GuardrailEvaluation | None = None,
) -> tuple[bool, list[SingleConditionResult]]:
    """Evaluates the 6 fundamental output testing conditions."""
    is_refusal = is_refusal_answer(answer)
    results: list[SingleConditionResult] = []

    # 1. Relevance to Question
    q_tokens = set(TOKEN_RE.findall(question.casefold())) - STOP_WORDS
    a_tokens = set(TOKEN_RE.findall(answer.casefold())) - STOP_WORDS
    overlap = len(q_tokens & a_tokens)
    relevance_score = (overlap / len(q_tokens)) if q_tokens else 1.0

    # Refusals for unavailable context or out-of-scope are considered relevant if they address refusal
    if not context_available and is_refusal:
        relevance_passed = True
        relevance_score = 1.0
        rel_desc = "Appropriately acknowledged and refused request."
    else:
        relevance_passed = relevance_score >= 0.15
        rel_desc = f"Overlap with question keywords: {overlap}/{len(q_tokens)} ({relevance_score * 100:.1f}%)."

    results.append(
        SingleConditionResult(
            condition=OutputTestCondition.RELEVANCE,
            passed=relevance_passed,
            score=round(relevance_score, 3),
            threshold=0.15,
            description=rel_desc,
            evidence=f"Question keywords: {sorted(list(q_tokens)[:5])}",
        )
    )

    # 2. Context Groundedness
    c_tokens = set(TOKEN_RE.findall(context.casefold())) - STOP_WORDS
    if is_refusal or not context_available:
        grounded_score = 1.0
        grounded_passed = True
        grounded_desc = "Refusal requires no grounding in document context."
    elif a_tokens:
        supported = len(a_tokens & c_tokens)
        grounded_score = supported / len(a_tokens)
        grounded_passed = grounded_score >= 0.20
        grounded_desc = f"Supported content tokens: {supported}/{len(a_tokens)} ({grounded_score * 100:.1f}%)."
    else:
        grounded_score = 1.0
        grounded_passed = True
        grounded_desc = "Answer contains no non-stopwords."

    results.append(
        SingleConditionResult(
            condition=OutputTestCondition.CONTEXT_GROUNDEDNESS,
            passed=grounded_passed,
            score=round(grounded_score, 3),
            threshold=0.20,
            description=grounded_desc,
            evidence=f"Grounding ratio: {grounded_score:.2f}",
        )
    )

    # 3. No Unsupported Claims
    if is_refusal or not context_available:
        claims_passed = True
        claims_score = 1.0
        claims_desc = "No factual claims asserted."
        unsupported = set()
    else:
        ans_nums = set(NUMBER_RE.findall(answer))
        ctx_nums = set(NUMBER_RE.findall(context))
        unsupported = {n for n in (ans_nums - ctx_nums) if not (n.isdigit() and int(n) in (1, 2, 3))}
        claims_passed = len(unsupported) == 0
        claims_score = 1.0 if claims_passed else 0.0
        claims_desc = "Zero unsupported numbers or dates." if claims_passed else f"Unsupported claims: {sorted(unsupported)}"

    results.append(
        SingleConditionResult(
            condition=OutputTestCondition.NO_UNSUPPORTED_CLAIMS,
            passed=claims_passed,
            score=claims_score,
            threshold=1.0,
            description=claims_desc,
            evidence=f"Unsupported entities: {sorted(unsupported) if unsupported else 'None'}",
        )
    )

    # 4. Format Compliance
    has_leak = bool(re.search(r"^(?:system|human|assistant|user|context):", answer, re.I | re.M))
    has_citation = bool(CITATION_RE.search(answer)) or is_refusal
    format_passed = (not has_leak) and has_citation
    format_score = 1.0 if format_passed else (0.5 if not has_leak else 0.0)

    results.append(
        SingleConditionResult(
            condition=OutputTestCondition.FORMAT_COMPLIANCE,
            passed=format_passed,
            score=format_score,
            threshold=1.0,
            description="Format compliant with bracketed citation." if format_passed else "Missing bracketed citation or prompt leakage detected.",
            evidence=f"Citation present: {has_citation}, Leak detected: {has_leak}",
        )
    )

    # 5. Answers When Sufficient Information Exists
    if context_available:
        ans_suff_passed = not is_refusal and len(answer.strip()) > 15
        ans_suff_desc = "Provided substantive answer based on available context." if ans_suff_passed else "Failed to answer despite available context (false refusal)."
    else:
        ans_suff_passed = True
        ans_suff_desc = "Context not available (condition not applicable, passed by default)."
    ans_suff_score = 1.0 if ans_suff_passed else 0.0

    results.append(
        SingleConditionResult(
            condition=OutputTestCondition.ANSWERS_WHEN_SUFFICIENT,
            passed=ans_suff_passed,
            score=ans_suff_score,
            threshold=1.0,
            description=ans_suff_desc,
            evidence=f"Context available: {context_available}, Is refusal: {is_refusal}",
        )
    )

    # 6. Appropriately Refuses When Information is Unavailable
    if not context_available:
        refuse_passed = is_refusal
        refuse_desc = "Appropriately refused when sufficient information was missing." if refuse_passed else "Failed to refuse: hallucinated an answer without evidence."
    else:
        refuse_passed = True
        refuse_desc = "Information was available (refusal not required, passed by default)."
    refuse_score = 1.0 if refuse_passed else 0.0

    results.append(
        SingleConditionResult(
            condition=OutputTestCondition.REFUSES_WHEN_UNAVAILABLE,
            passed=refuse_passed,
            score=refuse_score,
            threshold=1.0,
            description=refuse_desc,
            evidence=f"Context unavailable: {not context_available}, Refusal triggered: {is_refusal}",
        )
    )

    all_passed = all(r.passed for r in results)
    return all_passed, results
