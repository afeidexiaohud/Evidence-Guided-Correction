from egip.human_metrics import matching, pair_metrics, status_agreement


def test_spans_match_one_to_one():
    assert len(matching([(0, 5), (0, 6)], [(0, 5)])) == 1
    result = pair_metrics([(0, 5), (0, 6)], [(0, 5)])
    assert result['span_prf_iou_0_5']['precision'] == .5
    assert result['counts'] == [5, 6, 5]


def test_status_agreement():
    a = {'a': 'CONFLICT', 'b': 'NO_CONFLICT'}
    assert status_agreement(a, a)['cohen_kappa'] == 1.0
