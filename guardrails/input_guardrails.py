"""Input guardrails for NoteBuddy: validates length, blocks prompt injections, and enforces scope."""

from __future__ import annotations

import re
import time
from guardrails.models import GuardrailAction, GuardrailCheckResult, GuardrailStage

MAX_INPUT_LENGTH = 800
MIN_INPUT_LENGTH = 3

# Prompt injection & jailbreak heuristics
INJECTION_PATTERNS = [
    re.compile(r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|above|system)\s+(?:instructions|prompts|rules|directives)\b", re.I),
    re.compile(r"\b(?:you\s+are\s+now|act\s+as)\s+(?:dan|unrestricted|an\s+unfiltered|jailbroken|godmode)\b", re.I),
    re.compile(r"\b(?:show|reveal|print|dump|repeat|leak)\s+(?:your\s+)?(?:system\s+prompt|initial\s+instructions|hidden\s+prompt)\b", re.I),
    re.compile(r"\b(?:bypass|disable|turn\s+off)\s+(?:all\s+)?(?:safety|content|security|guardrail)\s+(?:filters|protocols|rules)\b", re.I),
    re.compile(r"(?:<\|im_start\|>|<\|system\|>|\[INST\]|<<SYS>>|system:)", re.I),
    re.compile(r"\b(?:from\s+now\s+on\s+you\s+must\s+never\s+refuse|do\s+anything\s+now)\b", re.I),
]

# Academic dishonesty & cheating bypasses
CHEATING_PATTERNS = [
    re.compile(r"\b(?:hack|bypass|exploit|cheat\s+on)\s+(?:the\s+)?(?:exam|quiz|proctoring|canvas|test|grade)\b", re.I),
    re.compile(r"\b(?:give\s+me\s+the\s+exact\s+answers\s+to\s+the\s+active\s+exam)\b", re.I),
]

# Explicit out-of-scope domains (NoteBuddy is for course notes, syllabus, policies & codebase)
OUT_OF_SCOPE_PATTERNS = [
    (re.compile(r"\b(?:cafeteria|canteen|mess|dining|menu|breakfast|lunch|dinner|pizza|burger|snack)\b", re.I), "Campus dining & cafeteria menus are outside course study scope."),
    (re.compile(r"\b(?:celebrity|hollywood|bollywood|taylor\s+swift|box\s+office|cricket|ipl|football|nba)\b", re.I), "Entertainment, celebrity, and sports trivia are outside course study scope."),
    (re.compile(r"\b(?:diagnose|medical|symptom|prescription|cure|headache|fever|cough|medicine|doctor|pharmacy)\b", re.I), "Medical and health queries are outside the application scope."),
    (re.compile(r"\b(?:stock|crypto|bitcoin|ethereum|shares|investment|trading)\b", re.I), "Financial investment advice is outside the application scope."),
    (re.compile(r"\b(?:relationship\s+advice|dating\s+tip|horoscope|zodiac|astrology)\b", re.I), "Personal lifestyle and astrology queries are outside the application scope."),
]


def check_input_length(question: str) -> GuardrailCheckResult:
    start = time.perf_counter()
    cleaned = question.strip()
    latency_ms = (time.perf_counter() - start) * 1000

    if not cleaned or len(cleaned) < MIN_INPUT_LENGTH:
        return GuardrailCheckResult(
            name="input_length",
            stage=GuardrailStage.INPUT,
            passed=False,
            action=GuardrailAction.REJECTED,
            reason=f"Question is too short or empty (minimum {MIN_INPUT_LENGTH} characters).",
            details={"length": len(cleaned), "min": MIN_INPUT_LENGTH, "max": MAX_INPUT_LENGTH},
            latency_ms=latency_ms,
        )

    if len(cleaned) > MAX_INPUT_LENGTH:
        return GuardrailCheckResult(
            name="input_length",
            stage=GuardrailStage.INPUT,
            passed=False,
            action=GuardrailAction.REJECTED,
            reason=f"Input exceeds maximum allowed length ({len(cleaned)}/{MAX_INPUT_LENGTH} characters). Excessively long inputs are restricted to prevent denial-of-service and context poisoning.",
            details={"length": len(cleaned), "max": MAX_INPUT_LENGTH},
            latency_ms=latency_ms,
        )

    return GuardrailCheckResult(
        name="input_length",
        stage=GuardrailStage.INPUT,
        passed=True,
        action=GuardrailAction.ALLOWED,
        reason="Input length is within acceptable bounds.",
        details={"length": len(cleaned)},
        latency_ms=latency_ms,
    )


def check_prompt_injection(question: str) -> GuardrailCheckResult:
    start = time.perf_counter()
    for pattern in INJECTION_PATTERNS:
        match = pattern.search(question)
        if match:
            return GuardrailCheckResult(
                name="prompt_injection",
                stage=GuardrailStage.INPUT,
                passed=False,
                action=GuardrailAction.REJECTED,
                reason=f"Potential prompt injection or jailbreak pattern detected: '{match.group(0)}'. Adversarial instruction overrides are blocked.",
                details={"matched_pattern": pattern.pattern, "matched_text": match.group(0)},
                latency_ms=(time.perf_counter() - start) * 1000,
            )

    for pattern in CHEATING_PATTERNS:
        match = pattern.search(question)
        if match:
            return GuardrailCheckResult(
                name="prompt_injection",
                stage=GuardrailStage.INPUT,
                passed=False,
                action=GuardrailAction.REJECTED,
                reason=f"Academic integrity violation: requests attempting to bypass exam proctoring or security are prohibited.",
                details={"matched_text": match.group(0)},
                latency_ms=(time.perf_counter() - start) * 1000,
            )

    return GuardrailCheckResult(
        name="prompt_injection",
        stage=GuardrailStage.INPUT,
        passed=True,
        action=GuardrailAction.ALLOWED,
        reason="No prompt injection or jailbreak patterns detected.",
        latency_ms=(time.perf_counter() - start) * 1000,
    )


def check_scope(question: str) -> GuardrailCheckResult:
    start = time.perf_counter()
    for pattern, description in OUT_OF_SCOPE_PATTERNS:
        match = pattern.search(question)
        if match:
            return GuardrailCheckResult(
                name="scope_enforcement",
                stage=GuardrailStage.INPUT,
                passed=False,
                action=GuardrailAction.INTERCEPTED_REFUSAL,
                reason=f"Out-of-scope query: {description}",
                details={"matched_category": match.group(0)},
                latency_ms=(time.perf_counter() - start) * 1000,
            )

    return GuardrailCheckResult(
        name="scope_enforcement",
        stage=GuardrailStage.INPUT,
        passed=True,
        action=GuardrailAction.ALLOWED,
        reason="Query is within academic course and repository study scope.",
        latency_ms=(time.perf_counter() - start) * 1000,
    )
