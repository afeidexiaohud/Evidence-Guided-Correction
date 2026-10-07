from __future__ import annotations
import argparse
from pathlib import Path
from egip.io import read_jsonl, write_jsonl
from egip.selection import select_many
from egip.utility import UtilityModel


def main():
    p = argparse.ArgumentParser(description='Select v4 final revisions from an aligned scored bank.')
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--ranked', type=Path, required=True)
    p.add_argument('--domain', choices=('fecdata', 'scifact'), required=True)
    p.add_argument('--model', type=Path, default=Path(__file__).resolve().parents[1] / 'resources/utility-v4.json')
    p.add_argument('--k', type=int, default=2)
    p.add_argument('--planner', choices=('egip', 'mmr'), default='egip')
    p.add_argument('--selector', choices=('full', 'base'), default='full')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    bank, ranks = read_jsonl(a.bank), read_jsonl(a.ranked)
    if [r['record_id'] for r in bank] != [r['record_id'] for r in ranks]:
        raise ValueError('Bank and ranked records must have identical ordered IDs')
    model = UtilityModel(a.model)
    rows = select_many(bank, ranks, model, a.domain, k=a.k, planner=a.planner, selector=a.selector)
    write_jsonl(a.output, rows)
    print(f'Selected {len(rows)} revisions: {a.output}')


if __name__ == '__main__':
    main()
