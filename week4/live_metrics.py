"""Reference-free live evidence checks, distinct from reference-based accuracy."""
from week4.metrics import TOKEN_RE, hallucination_proxy, relevance_f1

STOP = set('a an the is are was were be to of in on for and or with what when where how does do it this that'.split())

def live_evidence(question: str, answer: str, sources: list) -> dict:
    context = '\n\n'.join(s.excerpt for s in sources)
    tokens = set(TOKEN_RE.findall(answer.casefold())) - STOP
    supported = set(TOKEN_RE.findall(context.casefold())) - STOP
    flagged, reasons = hallucination_proxy(answer, context)
    return {
        'grading': 'Live evidence checks; add an expected answer to measure reference-based accuracy.',
        'question_overlap_f1': relevance_f1(answer, question),
        'context_token_support': len(tokens & supported) / len(tokens) if tokens and sources else None,
        'hallucination_flag': flagged if sources else None,
        'hallucination_reasons': reasons if sources else [],
        'retrieval_similarity_mean': sum(s.relevance_score for s in sources) / len(sources) if sources else None,
        'source_count': len(sources),
        'definitions': 'Question overlap is lexical token-set F1. Context support is the fraction of non-stopword answer tokens present in retrieved text; it does not check entailment. Similarity is the retriever score, not precision. Hallucination checks flag unsupported numbers and file paths only. Accuracy and retrieval recall require reference labels.',
    }
