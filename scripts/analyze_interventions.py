"""Table 3 / Appendices A-B diagnostics without additional generator calls."""
import argparse
import csv
from itertools import combinations
from pathlib import Path
from egip.io import read_jsonl, write_json, normalize
from egip.selection import validate_bank


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ranked', type=Path, required=True)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--k', type=int, default=2)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.k < 1:
        p.error('K must be positive')
    ranks, bank = read_jsonl(a.ranked), read_jsonl(a.bank)
    if len(ranks) != len(bank):
        raise ValueError('Bank/ranked population mismatch')
    counts = {'N': len(bank), 'K': a.k, 'same_plan': 0, 'partial_plan_overlap': 0,
              'disjoint_plans': 0, 'shared_realized_revision': 0}
    for name in ('egip', 'mmr'):
        counts.update({f'{name}_{metric}': 0 for metric in ('nested_samples', 'identical_span_samples', 'duplicate_revision_samples', 'short_budget_samples')})
    for item, rank in zip(bank, ranks):
        validate_bank(item, rank)
        ids = {name: item['planner_strategies'][name][:a.k] for name in ('egip', 'mmr')}
        sets = {name: set(values) for name, values in ids.items()}
        if sets['egip'] == sets['mmr']:
            counts['same_plan'] += 1
        elif sets['egip'] & sets['mmr']:
            counts['partial_plan_overlap'] += 1
        else:
            counts['disjoint_plans'] += 1
        mapping = {cid: index for index, cid in enumerate(item['mask_candidate_ids'])}
        texts = {name: [normalize(item['corrected sentences'][mapping[cid]]) for cid in ids[name]] for name in ids}
        counts['shared_realized_revision'] += bool(set(texts['egip']) & set(texts['mmr']))
        for name, chosen in ids.items():
            spans = [(rank['candidates'][cid]['start'], rank['candidates'][cid]['end']) for cid in chosen]
            nested = any((l <= r and e >= f) or (r <= l and f >= e) for (l, e), (r, f) in combinations(spans, 2))
            counts[f'{name}_nested_samples'] += nested
            counts[f'{name}_identical_span_samples'] += len(spans) != len(set(spans))
            counts[f'{name}_duplicate_revision_samples'] += len(texts[name]) != len(set(texts[name]))
            counts[f'{name}_short_budget_samples'] += len(chosen) < a.k
    counts['rates'] = {name: value / max(1, counts['N']) for name, value in counts.items() if name not in ('N', 'K')}
    write_json(a.output, counts)
    print(counts)


if __name__ == '__main__':
    main()
