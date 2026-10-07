"""Replay all four v4 factorial cells at K=1..10 on one immutable shared bank."""
import argparse
import csv
from pathlib import Path
from egip.io import read_jsonl, write_jsonl
from egip.evaluation import evaluate
from egip.selection import select_many
from egip.utility import UtilityModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bank', 'ranked', 'canonical', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--domain', choices=('fecdata', 'scifact'), required=True)
    p.add_argument('--model', type=Path, default=Path(__file__).resolve().parents[1] / 'resources/utility-v4.json')
    a = p.parse_args()
    bank, ranked, canonical = map(read_jsonl, (a.bank, a.ranked, a.canonical))
    if [r['record_id'] for r in bank] != [r['record_id'] for r in canonical]:
        raise ValueError('Bank/canonical ordered IDs differ')
    if any(len(r['planner_strategies'][name]) != min(10, len(q['candidates']))
           for r, q in zip(bank, ranked) for name in ('egip', 'mmr')):
        raise ValueError('K=1..10 replay requires the complete top-10 bank for both planners')
    model = UtilityModel(a.model)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(1, 11):
        for planner in ('egip', 'mmr'):
            for selector in ('full', 'base'):
                predictions = select_many(bank, ranked, model, a.domain, k=k, planner=planner, selector=selector)
                metric = evaluate(canonical, predictions, 3882 if a.domain == 'fecdata' else 100)
                rows.append({'dataset': a.domain, 'K': k, 'planner': planner.upper(), 'selector': selector, **metric})
                write_jsonl(a.output_dir / f'{a.domain}.k{k}.{planner}.{selector}.jsonl', predictions)
    with (a.output_dir / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'Replayed 40 cells for {a.domain}')


if __name__ == '__main__':
    main()
