import argparse
from pathlib import Path
from egip.evaluation import evaluate
from egip.io import read_jsonl, write_json, sha256


def main():
    p = argparse.ArgumentParser(description='Strict v4 canonical SARI/Exact evaluation.')
    p.add_argument('--canonical', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--expected-n', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = evaluate(read_jsonl(a.canonical), read_jsonl(a.predictions), a.expected_n)
    result.update(canonical_sha256=sha256(a.canonical), prediction_sha256=sha256(a.predictions))
    write_json(a.output, result)
    print(result)


if __name__ == '__main__':
    main()
