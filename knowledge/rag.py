"""Retrieval-augmented answer orchestration for Exercise 3."""

from app.ollama_client import OllamaClient
from knowledge.models import RetrievedChunk
from knowledge.store import KnowledgeStore


NO_CONTEXT_ANSWER = (
    "I couldn't find relevant information in the indexed course materials."
)


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


class RagService:
    def __init__(
        self,
        store: KnowledgeStore,
        llm: OllamaClient,
        top_k: int = 4,
    ) -> None:
        self.store = store
        self.llm = llm
        self.top_k = top_k

    async def answer(self, question: str) -> tuple[str, list[RetrievedChunk]]:
        sources = self.store.query(question, self.top_k)
        if not sources:
            return NO_CONTEXT_ANSWER, []
        prompt = build_grounded_prompt(question, sources)
        generated = await self.llm.generate(prompt)
        context_text = "\n\n".join(source.text for source in sources)
        if _is_refusal(generated) or not _is_grounded_in_context(generated, context_text):
            return NO_CONTEXT_ANSWER, []
        return ensure_citation(generated, sources), sources


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


def ensure_citation(answer: str, sources: list[RetrievedChunk]) -> str:
    """Guarantee at least one exact citation even when the model omits formatting."""
    if _is_refusal(answer):
        return answer
    if any(f"[{source.citation}]" in answer for source in sources):
        return answer
    return f"{answer.rstrip()} [{sources[0].citation}]"


def build_grounded_prompt(question: str, sources: list[RetrievedChunk]) -> str:
    context_blocks = []
    for index, source in enumerate(sources, start=1):
        context_blocks.append(
            f"[Source {index}: {source.citation}]\n{source.text}"
        )
    context = "\n\n".join(context_blocks)
    return f"""You are NoteBuddy, a course study assistant. Answer the student's question using ONLY the supplied course context.

CRITICAL RULES:
1. Answer the student's question using ONLY the supplied course context. Under NO circumstances should you use outside knowledge, general trivia, world events, or unmentioned facts.
2. If the context does not explicitly contain the answer, or if the question is unrelated to the context, you MUST say exactly: {NO_CONTEXT_ANSWER}
3. Cite every factual claim inline using the exact citation label in brackets, for example [example-syllabus.md, section: Assessment schedule].
4. Be clear and concise.

COURSE CONTEXT
{context}

STUDENT QUESTION
{question}

GROUNDED ANSWER
"""
