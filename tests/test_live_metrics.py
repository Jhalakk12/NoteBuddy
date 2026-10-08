from week4.live_metrics import live_evidence
from services.contracts import SourceDto

def test_custom_question_gets_evidence_without_inventing_accuracy():
    source = SourceDto(citation='notes', source='notes', page=None, section='', excerpt='Assignment 2 is due October 9.', relevance_score=.7)
    metrics = live_evidence('When is Assignment 2 due?', 'Assignment 2 is due October 10.', [source])
    assert 0 < metrics['context_token_support'] < 1
    assert metrics['hallucination_flag'] is True
    assert metrics['retrieval_similarity_mean'] == .7
    assert 'accuracy' not in metrics

def test_no_context_does_not_claim_hallucination_score():
    metrics = live_evidence('Write a function', 'def f(): return 42', [])
    assert metrics['context_token_support'] is None
    assert metrics['hallucination_flag'] is None
