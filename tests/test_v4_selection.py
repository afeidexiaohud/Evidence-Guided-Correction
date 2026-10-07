from pathlib import Path
import numpy as np
import pytest
from egip.selection import select_many, select_revision, use_base_branch
from egip.utility import UtilityModel, make_features
from egip.evaluation import evaluate
from egip.candidates import Candidate, mmr_rank


def example():
    claim = 'alpha not beta'
    candidates = [{'start': 0, 'end': 5, 'text': 'alpha', 'phrase': 'alpha', 'error_probability': .7, 'relevance': .2},
                  {'start': 0, 'end': 9, 'text': 'alpha not', 'phrase': 'alpha not', 'error_probability': .8, 'relevance': .3}]
    rank = {'record_id': 'fixture:1', 'input_claim': claim, 'evidence': ['alpha beta'], 'candidates': candidates}
    bank = {'record_id': 'fixture:1', 'input_claim': claim, 'evidence': ['alpha beta'], 'mask_candidate_ids': [0, 1],
            'corrected sentences': ['alpha beta', 'alpha beta', claim], 'correction_scores': [.2, .2, .2],
            'rouge_scores': [.1, .1, .1], 'planner_strategies': {'egip': [0, 1], 'mmr': [1, 0]}}
    return bank, rank


def test_duplicate_revisions_count_and_first_max_wins():
    bank, rank = example()
    model = UtilityModel(Path(__file__).resolve().parents[1] / 'resources/utility-v4.json')
    for planner in ('egip', 'mmr'):
        row = select_revision(bank, rank, model, 'fecdata', planner=planner, selector='base')
        assert row['effective_K'] == 2
        assert row['selected_candidate_id'] == bank['planner_strategies'][planner][0]
        assert row == select_many([bank], [rank], model, 'fecdata', planner=planner, selector='base')[0]
    bank['correction_scores'][-1] = 1.0
    assert select_revision(bank, rank, model, 'fecdata', selector='base')['input_fallback']


def test_feature_position_not_clipped_and_no_reference_input():
    bank, rank = example()
    a = make_features(bank, 0, 10, rank['candidates'][0], 'fecdata')
    bank['gt_claim'] = 'an irrelevant test reference'
    assert a == make_features(bank, 0, 10, rank['candidates'][0], 'fecdata')
    assert a[10] == 2.0 and len(a) == 19


def test_gate_strict_threshold_and_negation():
    bank, _ = example()
    bank['input_claim'] = 'activation'
    bank['evidence'] = [' '.join(['activation'] * 100)]
    assert not use_base_branch(bank)
    bank['evidence'] = [' '.join(['activation'] * 101)]
    assert use_base_branch(bank)
    bank['input_claim'] = 'not true: activation'
    assert not use_base_branch(bank)


def test_hybrid_differs_from_pure_q_and_ties_are_stable():
    c = [Candidate(i, i + 1, 'x', 'x') for i in range(3)]
    doc = np.array([1., 0.])
    emb = np.array([[1., 0.], [0., 1.], [0., 1.]])
    order, _ = mmr_rank(c, doc, emb, error_probabilities=np.array([.80, .81, .81]), error_weight=.9, top_k=3)
    assert order == [0, 1, 2]


def test_strict_canonical_alignment_and_empty_prediction():
    refs = [{'record_id': 'a', 'input_claim': 'a b', 'reference': 'a c'},
            {'record_id': 'b', 'input_claim': 'a', 'reference': 'a'}]
    preds = [{'record_id': 'a', 'prediction': ' a   c '}, {'record_id': 'b', 'prediction': 'a'}]
    assert evaluate(refs, preds, 2)['exact_count'] == 2
    with pytest.raises(ValueError, match='reordered'):
        evaluate(refs, preds[::-1], 2)
    preds[0]['prediction'] = ' '
    with pytest.raises(ValueError, match='Empty'):
        evaluate(refs, preds, 2)


def test_bank_fails_on_missing_attempt_score():
    bank, rank = example()
    bank['rouge_scores'].pop()
    model = UtilityModel(Path(__file__).resolve().parents[1] / 'resources/utility-v4.json')
    with pytest.raises(ValueError, match='rouge_scores'):
        select_many([bank], [rank], model, 'fecdata')
