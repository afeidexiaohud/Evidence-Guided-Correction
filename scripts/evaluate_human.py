"""Evaluate independently locked human spans or annotator agreement."""
import argparse
import json
from pathlib import Path
from egip.human_metrics import evaluate, status_agreement, spans, pair_metrics, prf
from egip.io import write_json


def read(path):
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if len(rows) != len({r['item_id'] for r in rows}):
        raise ValueError('Duplicate annotation IDs')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--predictions', type=Path)
    p.add_argument('--annotator-b', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if (a.predictions is None) == (a.annotator_b is None):
        p.error('Choose predictions or a paired second annotator')
    labels = read(a.labels)
    if a.predictions:
        predictions = {r['item_id']: r['spans'] for r in read(a.predictions)}
        result = {dataset: evaluate([r for r in labels if r['dataset'] == dataset],
                  {r['item_id']: predictions[r['item_id']] for r in labels if r['dataset'] == dataset})
                  for dataset in sorted({r['dataset'] for r in labels})}
        if set(predictions) != {r['item_id'] for r in labels}:
            raise ValueError('Annotation/prediction ID mismatch')
    else:
        second = {r['item_id']: r for r in read(a.annotator_b)}
        result = {'status': status_agreement({r['item_id']: r['factual_status'] for r in labels},
                                             {rid: r['factual_status'] for rid, r in second.items()})}
        counts = [0, 0, 0]
        span_counts = [0, 0, 0]
        included = 0
        for row in labels:
            other = second[row['item_id']]
            if row['claim'] != other['claim']:
                raise ValueError('Annotator claim mismatch')
            if row['factual_status'] == other['factual_status'] == 'CONFLICT' and row['spans'] and other['spans']:
                metric = pair_metrics(spans(row['claim'], row['spans']), spans(other['claim'], other['spans']))
                counts = [a + b for a, b in zip(counts, metric['counts'])]
                span_counts = [a + b for a, b in zip(span_counts,
                              [metric['matched_spans'], metric['predicted_spans'], metric['gold_spans']])]
                included += 1
        result.update(both_conflict_localizable=included, character_micro=prf(*counts), span_iou_0_5_micro=prf(*span_counts))
    write_json(a.output, result)


if __name__ == '__main__':
    main()
