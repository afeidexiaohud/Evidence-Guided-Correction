"""Rebuild the exact canonical benchmark files from upstream source bytes."""
import argparse
import hashlib
import json
from pathlib import Path
from egip.io import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain', choices=('fecdata', 'scifact'), required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-different-source', action='store_true', help='New experiment only; disables frozen hash parity.')
    a = p.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'resources/datasets-v4.json').read_text())[a.domain]
    if not a.allow_different_source and sha256(a.source) != manifest['source_sha256']:
        raise ValueError('Source hash differs from the frozen v4 input')
    rows = []
    for index, raw in enumerate(a.source.read_bytes().splitlines()):
        if not raw.strip():
            raise ValueError('Blank source rows change canonical identity')
        item = json.loads(raw)
        rid = (f"fecdata-test:{item['claim_id']}" if a.domain == 'fecdata' else
               f"scifact-fec-test:{index:04d}:{hashlib.sha256(raw).hexdigest()[:12]}")
        rows.append({'record_id': rid, 'input_claim': item[manifest['input_source_field']],
                     'evidence': item[manifest['evidence_source_field']], 'reference': item[manifest['reference_source_field']]})
    if len({r['record_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate canonical IDs')
    if not a.allow_different_source and len(rows) != manifest['record_count']:
        raise ValueError('Unexpected benchmark size')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = ''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows)
    if not a.allow_different_source and hashlib.sha256(encoded.encode()).hexdigest() != manifest['canonical_sha256']:
        raise ValueError('Canonical byte parity failed')
    a.output.write_text(encoded)
    write_json(a.output.with_suffix('.manifest.json'), {'domain': a.domain, 'N': len(rows),
               'source_sha256': sha256(a.source), 'canonical_sha256': sha256(a.output),
               'frozen_v4': not a.allow_different_source})
    print(f'Prepared {len(rows)} canonical records: {a.output}')


if __name__ == '__main__':
    main()
