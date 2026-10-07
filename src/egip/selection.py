"""v4 utility selection, base selection, and frozen context routing."""
from __future__ import annotations
import re
import numpy as np
from .io import normalize
from .utility import UtilityModel, make_features


def gate_tokens(text):
    return re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)?', str(text).lower())


def technical_density(text):
    tokens = gate_tokens(text)
    suffixes = ('ase', 'osis', 'itis', 'genic', 'tion', 'tive', 'emia', 'acid', 'cell', 'gene', 'protein')
    return sum(len(t) >= 9 or t.endswith(suffixes) or any(c.isdigit() for c in t)
               for t in tokens) / max(1, len(tokens))


def use_base_branch(item):
    claim = str(item['input_claim'])
    evidence = ' '.join(map(str, item['evidence']))
    negation = bool(re.search(r'\b(not|no|never|none|without)\b', claim.lower())) or claim.lower().startswith('not true:')
    return len(gate_tokens(evidence)) / 300 > 1 / 3 and technical_density(claim + ' ' + evidence) > 0.18421052631578946 and not negation


def validate_bank(item, ranked):
    if str(item['record_id']) != str(ranked['record_id']) or item['input_claim'] != ranked['input_claim']:
        raise ValueError('Bank/ranked identity or source mismatch')
    if ' '.join(map(str, item['evidence'])) != ' '.join(map(str, ranked['evidence'][:3])):
        raise ValueError('Bank evidence must match the first three ranked evidence units when joined')
    texts = item['corrected sentences']
    ids = item['mask_candidate_ids']
    if len(ids) != len(set(ids)) or len(texts) != len(ids) + 1:
        raise ValueError('Bank must contain one output per mask and a final free fallback')
    if any(not normalize(text) for text in texts):
        raise ValueError('Empty revision in bank')
    if normalize(texts[-1]) != normalize(item['input_claim']):
        raise ValueError('Final bank entry must be the unchanged source')
    for name in ('correction_scores', 'rouge_scores'):
        if len(item[name]) != len(texts) or not np.isfinite(item[name]).all():
            raise ValueError(f'Invalid aligned {name}')
    available = set(ids)
    for name in ('egip', 'mmr'):
        order = item['planner_strategies'][name]
        if len(order) != len(set(order)) or not set(order) <= available:
            raise ValueError('Planner order contains duplicated or ungenerated candidate IDs')
    for cid in ids:
        if not 0 <= cid < len(ranked['candidates']):
            raise ValueError('Candidate ID is outside the ranked pool')


def select_revision(item, ranked, model: UtilityModel, domain, *, k=2, planner='egip', selector='full'):
    if k < 1 or planner not in ('egip', 'mmr') or selector not in ('full', 'base'):
        raise ValueError('Invalid budget/planner/selector')
    validate_bank(item, ranked)
    ids = list(item['planner_strategies'][planner][:k])
    mapping = {cid: i for i, cid in enumerate(item['mask_candidate_ids'])}
    indices = [mapping[cid] for cid in ids] + [len(item['corrected sentences']) - 1]
    cids = ids + [None]
    features = [make_features(item, oi, pos, ranked['candidates'][cid] if cid is not None else None, domain)
                for pos, (oi, cid) in enumerate(zip(indices, cids))]
    base_index = int(np.argmax([item['correction_scores'][i] + item['rouge_scores'][i] for i in indices]))
    base = selector == 'base' or use_base_branch(item)
    selected = base_index if base else int(np.argmax(model.predict(features, domain)))
    output_index = indices[selected]
    return {'record_id': str(item['record_id']), 'prediction': normalize(item['corrected sentences'][output_index]),
            'K': k, 'effective_K': len(ids), 'planner': planner, 'selector': selector,
            'selection_branch': 'base' if base else 'utility', 'selected_candidate_id': cids[selected],
            'selected_position': selected, 'input_fallback': cids[selected] is None}


def select_many(bank, ranked, model: UtilityModel, domain, *, k=2, planner='egip', selector='full'):
    """Batch utility prediction without changing per-record argmax/tie order."""
    if k < 1 or planner not in ('egip', 'mmr') or selector not in ('full', 'base'):
        raise ValueError('Invalid budget/planner/selector')
    if len(bank) != len(ranked):
        raise ValueError('Bank/ranked population mismatch')
    contexts, features = [], []
    for item, rank in zip(bank, ranked):
        validate_bank(item, rank)
        ids = list(item['planner_strategies'][planner][:k])
        mapping = {cid: i for i, cid in enumerate(item['mask_candidate_ids'])}
        indices = [mapping[cid] for cid in ids] + [len(item['corrected sentences']) - 1]
        cids = ids + [None]
        start = len(features)
        features.extend(make_features(item, oi, pos, rank['candidates'][cid] if cid is not None else None, domain)
                        for pos, (oi, cid) in enumerate(zip(indices, cids)))
        contexts.append((item, ids, cids, indices, start, len(features)))
    if not contexts:
        return []
    scores = model.predict(features, domain) if selector == 'full' else None
    result = []
    for item, ids, cids, indices, start, end in contexts:
        base = selector == 'base' or use_base_branch(item)
        chosen = (int(np.argmax([item['correction_scores'][i] + item['rouge_scores'][i] for i in indices]))
                  if base else int(np.argmax(scores[start:end])))
        result.append({'record_id': str(item['record_id']),
            'prediction': normalize(item['corrected sentences'][indices[chosen]]), 'K': k,
            'effective_K': len(ids), 'planner': planner, 'selector': selector,
            'selection_branch': 'base' if base else 'utility', 'selected_candidate_id': cids[chosen],
            'selected_position': chosen, 'input_fallback': cids[chosen] is None})
    return result
