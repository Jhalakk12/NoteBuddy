"""Application-level coordination across Retrieval/RAG and LLM services."""

from knowledge.rag import NO_CONTEXT_ANSWER
from services.application.client import ServiceClient
from services.contracts import OrchestratedAnswer, RawAnswer


import re

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*", re.I)
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on", "for",
    "and", "or", "with", "what", "when", "where", "how", "does", "do", "it", "this", "that",
    "not", "by", "as", "at", "from", "which", "will", "can", "has", "have", "had",
}


def _is_grounded_in_context(answer: str, context: str, threshold: float = 0.20) -> bool:
    tokens = set(TOKEN_RE.findall(answer.casefold())) - STOP_WORDS
    if not tokens:
        return True
    supported = set(TOKEN_RE.findall(context.casefold())) - STOP_WORDS
    return (len(tokens & supported) / len(tokens)) >= threshold


from guardrails.engine import GuardrailEngine
from guardrails.models import GuardrailAction, GuardrailEvaluation


class Orchestrator:
    def __init__(self, client: ServiceClient, top_k: int = 4) -> None:
        self.client = client
        self.top_k = top_k

    async def raw_answer(self, question: str) -> RawAnswer:
        generated = await self.client.generate(question)
        return RawAnswer(answer=generated.answer, model=generated.model)

    async def grounded_answer(self, question: str, with_guardrails: bool = True) -> OrchestratedAnswer:
        steps = ["application"]

        # Stage 1: Input Guardrails
        if with_guardrails:
            input_eval = GuardrailEngine.evaluate_input(question)
            if not input_eval.passed:
                return OrchestratedAnswer(
                    answer=input_eval.sanitized_output or NO_CONTEXT_ANSWER,
                    sources=[],
                    model="not-called",
                    orchestration=steps,
                    guardrail=input_eval.model_dump(),
                )

        # Stage 2: Retrieval
        retrieved = await self.client.retrieve(question, self.top_k)
        steps.append("retrieval")

        if with_guardrails:
            retrieval_eval = GuardrailEngine.evaluate_retrieval(retrieved.sources)
            if not retrieval_eval.passed:
                return OrchestratedAnswer(
                    answer=retrieval_eval.sanitized_output or NO_CONTEXT_ANSWER,
                    sources=[],
                    model="not-called",
                    orchestration=steps,
                    guardrail=retrieval_eval.model_dump(),
                )
        else:
            if not retrieved.sources or not retrieved.grounded_prompt:
                return OrchestratedAnswer(
                    answer=NO_CONTEXT_ANSWER,
                    sources=[],
                    model="not-called",
                    orchestration=steps,
                    guardrail={"passed": False, "mode": "disabled"},
                )

        # Stage 3: LLM Generation
        prompt = retrieved.grounded_prompt or question
        generated = await self.client.generate(prompt)
        steps.append("llm")
        context_text = "\n\n".join(source.excerpt for source in retrieved.sources)

        # Stage 4: Output Guardrails
        if with_guardrails:
            is_ref = _is_refusal(generated.answer)
            output_eval = GuardrailEngine.evaluate_output(generated.answer, context_text, is_ref)
            if not output_eval.passed:
                return OrchestratedAnswer(
                    answer=output_eval.sanitized_output or NO_CONTEXT_ANSWER,
                    sources=[],
                    model=generated.model,
                    orchestration=steps,
                    guardrail=output_eval.model_dump(),
                )
            clean_answer = output_eval.sanitized_output or generated.answer
            final_answer = _ensure_citation(clean_answer, retrieved.sources[0].citation) if retrieved.sources else clean_answer
            return OrchestratedAnswer(
                answer=final_answer,
                sources=retrieved.sources,
                model=generated.model,
                orchestration=steps,
                guardrail=output_eval.model_dump(),
            )

        # Without Guardrails: Baseline unverified generation
        if _is_refusal(generated.answer) or not _is_grounded_in_context(generated.answer, context_text):
            return OrchestratedAnswer(
                answer=generated.answer,
                sources=retrieved.sources,
                model=generated.model,
                orchestration=steps,
                guardrail={"passed": False, "mode": "disabled"},
            )

        answer = _ensure_citation(generated.answer, retrieved.sources[0].citation) if retrieved.sources else generated.answer
        return OrchestratedAnswer(
            answer=answer,
            sources=retrieved.sources,
            model=generated.model,
            orchestration=steps,
            guardrail={"passed": True, "mode": "disabled"},
        )


def _is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return (
        NO_CONTEXT_ANSWER.lower() in lowered
        or "couldn't find relevant information" in lowered
        or "could not find relevant information" in lowered
        or "cannot find relevant information" in lowered
        or "do not contain" in lowered
        or "does not contain" in lowered
        or "not provided" in lowered
        or "insufficient" in lowered
        or "unable to find" in lowered
    )


def _ensure_citation(answer: str, primary_citation: str) -> str:
    if _is_refusal(answer):
        return answer
    if f"[{primary_citation}]" in answer:
        return answer
    return f"{answer.rstrip()} [{primary_citation}]"

