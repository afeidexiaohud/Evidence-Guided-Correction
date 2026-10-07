"""Frozen v4 hybrid EGIP order and matched MMR, with deterministic ties."""
import argparse
from pathlib import Path
import numpy as np
from egip.candidates import Candidate, mmr_rank
from egip.io import read_jsonl, write_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--encoder', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--revision', default='c9745ed1d9f207416be6d2e6f8de32d1f16199bf')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=256)
    a = p.parse_args()
    from sentence_transformers import SentenceTransformer
    rows = read_jsonl(a.input)
    model = SentenceTransformer(a.encoder, revision=a.revision, device=a.device)
    texts, slices = [], []
    for row in rows:
        start = len(texts)
        texts.extend([str(row['input_claim'])] + [str(c['phrase']) for c in row['candidates']])
        slices.append((start, len(texts)))
    embeddings = model.encode(texts, batch_size=a.batch_size, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=True)
    for row, (start, end) in zip(rows, slices):
        candidates = [Candidate(c['start'], c['end'], c['text'], c['phrase']) for c in row['candidates']]
        q = np.asarray([c['error_probability'] for c in row['candidates']])
        kwargs = dict(top_k=len(candidates), diversity=0.3)
        order, relevance = mmr_rank(candidates, embeddings[start], embeddings[start + 1:end],
                                    error_probabilities=q, error_weight=0.9, **kwargs)
        mmr, _ = mmr_rank(candidates, embeddings[start], embeddings[start + 1:end], **kwargs)
        row['egip_rank'], row['mmr_rank'] = order, mmr
        for candidate, value in zip(row['candidates'], relevance):
            candidate['relevance'] = float(value)
    write_jsonl(a.output, rows)
    print(f'Ranked {len(rows)} records: {a.output}')


if __name__ == '__main__':
    main()
