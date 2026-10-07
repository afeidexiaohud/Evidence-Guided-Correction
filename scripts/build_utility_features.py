"""Construct K=5 utility targets from development banks, references and split IDs."""
import argparse
from pathlib import Path
from egip.evaluation import sari
from egip.io import read_jsonl, write_jsonl
from egip.selection import validate_bank
from egip.utility import make_features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bank', 'ranked', 'references', 'splits', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--domain', choices=('fecdata', 'scifact'), required=True)
    a = p.parse_args()
    bank, ranked = read_jsonl(a.bank), read_jsonl(a.ranked)
    references = {r['record_id']: r['reference'] for r in read_jsonl(a.references)}
    splits = {r['record_id']: r['split'] for r in read_jsonl(a.splits)}
    if [r['record_id'] for r in bank] != [r['record_id'] for r in ranked] or set(splits) != set(references):
        raise ValueError('Development record identities differ')
    rows = []
    for item, rank in zip(bank, ranked):
        validate_bank(item, rank)
        rid = item['record_id']
        if rid not in splits or splits[rid] not in ('train', 'guard', 'holdout') or '-test:' in rid:
            raise ValueError('Only the fixed development partitions may be labeled for utility training')
        ids = item['planner_strategies']['egip'][:5]
        mapping = {cid: i for i, cid in enumerate(item['mask_candidate_ids'])}
        indices = [mapping[cid] for cid in ids] + [len(item['corrected sentences']) - 1]
        candidates = [rank['candidates'][cid] for cid in ids] + [None]
        predictions = [str(item['corrected sentences'][i]).strip() for i in indices]
        reference = references[rid]
        rows.append({'record_id': rid, 'domain': a.domain, 'split': splits[rid],
            'features': [make_features(item, i, pos, c, a.domain) for pos, (i, c) in enumerate(zip(indices, candidates))],
            'sari': [sari(item['input_claim'], text, reference) / 100 for text in predictions],
            'exact': [float(text == reference.strip()) for text in predictions],
            'changed': [text != item['input_claim'].strip() for text in predictions]})
    write_jsonl(a.output, rows)


if __name__ == '__main__':
    main()
