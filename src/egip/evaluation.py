"""Original per-example FECDATA SARI and normalized Exact, reported in percent."""
from __future__ import annotations
import numpy as np
from ._vendor.sari import SARIsent
from .io import normalize


def sari(source, prediction, reference):
    return float(SARIsent(normalize(source), normalize(prediction), [normalize(reference)])[3]) * 100


def evaluate(canonical, predictions, expected_n=None):
    if not canonical or (expected_n is not None and len(canonical) != expected_n):
        raise ValueError('Unexpected canonical population size')
    ids = [str(r['record_id']) for r in canonical]
    pids = [str(r['record_id']) for r in predictions]
    if len(ids) != len(set(ids)) or pids != ids:
        raise ValueError('Missing, duplicate, extra, or reordered prediction IDs')
    scores = []
    exact = 0
    for ref, pred in zip(canonical, predictions):
        text = normalize(pred['prediction'])
        if not text:
            raise ValueError(f'Empty prediction: {ref["record_id"]}')
        scores.append(sari(ref['input_claim'], text, ref['reference']))
        exact += text == normalize(ref['reference'])
    return {'N': len(ids), 'SARI': float(np.mean(scores)),
            'Exact_percent': 100 * exact / len(ids), 'exact_count': exact}
