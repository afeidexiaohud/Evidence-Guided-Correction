"""Generate independent original-claim attempts and score the shared bank."""
from __future__ import annotations
import argparse
import importlib.util
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from egip.backend import validate_backend
from egip.candidates import Candidate, apply_candidate_mask
from egip.io import read_jsonl, write_jsonl, write_json, sha256


def prepare_bank(ranked, k):
    bank = {}
    for index, row in enumerate(ranked):
        candidates = row['candidates']
        orders = {name: list(row[name + '_rank'][:k]) for name in ('egip', 'mmr')}
        for order in orders.values():
            if len(order) != len(set(order)) or any(type(i) is not int or not 0 <= i < len(candidates) for i in order):
                raise ValueError('Invalid candidate order')
        ids = list(dict.fromkeys(orders['egip'] + orders['mmr']))
        masked = [apply_candidate_mask(row['input_claim'], Candidate(**{key: candidates[cid][key] for key in ('start', 'end', 'text', 'phrase')})) for cid in ids]
        bank[index] = {'record_id': row['record_id'], 'id': index, 'input_claim': row['input_claim'],
                       'evidence': list(row['evidence'][:3]), 'masked_claim': masked,
                       'mask_candidate_ids': ids, 'planner_strategies': orders}
    return bank


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ranked', type=Path, required=True)
    p.add_argument('--domain', choices=('fecdata', 'scifact'), required=True)
    p.add_argument('--backend', type=Path, default=Path('third_party/MaskToCorrect'))
    p.add_argument('--model', required=True)
    p.add_argument('--docnli', type=Path, required=True)
    p.add_argument('--roberta', required=True)
    p.add_argument('--scorer-python', default=sys.executable)
    p.add_argument('--k', type=int, default=2)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--work-dir', type=Path, required=True)
    a = p.parse_args()
    if not 1 <= a.k <= 10:
        p.error('v4 budget is between 1 and 10')
    backend = validate_backend(a.backend)
    ranked = read_jsonl(a.ranked)
    bank = prepare_bank(ranked, a.k)
    a.work_dir.mkdir(parents=True, exist_ok=True)
    # External source preserves the exact raw prompt, sampling settings and parser.
    from vllm import LLM
    spec = importlib.util.spec_from_file_location('egip_shared_corrector', backend / 'corrector/corrected_candidates.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    llm = LLM(model=a.model, tensor_parallel_size=2, max_model_len=4096)
    corrected = module.Corrector().correct(bank, llm, SimpleNamespace(model='qwen', shots=3,
                                           dataset='fever' if a.domain == 'fecdata' else 'scifact'))
    for item in corrected.values():
        if len(item['corrected sentences']) != len(item['mask_candidate_ids']) or any(not str(x).strip() for x in item['corrected sentences']):
            raise ValueError('Empty or unaligned generation; refusing to silently drop an attempt')
    # Score in a separate process/environment; release generator GPU allocations.
    del llm
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    bank_path = a.work_dir / 'generated.pkl'
    with bank_path.open('wb') as handle:
        pickle.dump(corrected, handle)
    environment = dict(os.environ, M2C_DOCNLI_PATH=str(a.docnli.resolve()), M2C_ROBERTA_PATH=a.roberta)
    subprocess.run([a.scorer_python, str(backend / 'correction_scoring.py'), str(bank_path.resolve())],
                   cwd=str(backend), env=environment, check=True)
    with bank_path.with_name('generated_cs.pkl').open('rb') as handle:
        scored = pickle.load(handle)
    from egip.selection import validate_bank
    for item, rank in zip(scored.values(), ranked):
        validate_bank(item, rank)
    write_jsonl(a.output, list(scored.values()))
    write_json(a.output.with_suffix('.manifest.json'), {'domain': a.domain, 'K': a.k,
        'ranked_sha256': sha256(a.ranked), 'bank_sha256': sha256(a.output),
        'docnli_sha256': sha256(a.docnli), 'generation': 'new; numerical identity with frozen outputs is not assumed'})
    print(f'Generated and scored {len(scored)} shared-bank records: {a.output}')


if __name__ == '__main__':
    main()
