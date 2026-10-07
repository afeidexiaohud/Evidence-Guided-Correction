"""Build the frozen v4 pool from canonical inputs; references are not carried forward."""
import argparse
from pathlib import Path
from egip.candidates import build_candidate_pool
from egip.io import read_jsonl, write_jsonl


def evidence_text(value):
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return f'{value[0]}\t{value[1]}'
    return str(value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--canonical', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = []
    for index, row in enumerate(read_jsonl(a.canonical)):
        claim = str(row['input_claim'])
        rows.append({'record_id': str(row['record_id']), 'id': index, 'input_claim': claim,
                     'evidence': [evidence_text(x) for x in row['evidence'][:3]],
                     'candidates': [c.to_dict() for c in build_candidate_pool(claim)]})
    write_jsonl(a.output, rows)
    print(f'Prepared {len(rows)} candidate pools: {a.output}')


if __name__ == '__main__':
    main()
