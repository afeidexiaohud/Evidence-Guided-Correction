"""Refit frozen v4 utility models, or rerun the original development-only search."""
import argparse
from pathlib import Path
import numpy as np
from egip.io import read_jsonl, write_json, sha256
from egip.utility import export_utility


def fit(rows, mode, exact_weight, depth, include_guard):
    from sklearn.ensemble import HistGradientBoostingRegressor
    domains = ('fecdata', 'scifact') if mode == 'per_domain' else ('combined',)
    models = {}
    for domain in domains:
        selected = [r for r in rows if r['split'] in ({'train', 'guard'} if include_guard else {'train'})
                    and (domain == 'combined' or r['domain'] == domain)]
        x = np.concatenate([r['features'] for r in selected])
        y = np.concatenate([np.asarray(r['sari']) + exact_weight * np.asarray(r['exact']) for r in selected])
        models[domain] = HistGradientBoostingRegressor(max_iter=60, learning_rate=0.07, max_depth=depth,
            min_samples_leaf=20, l2_regularization=1.0, random_state=20260923).fit(x, y)
    return models


def report(rows, models, split, domain):
    chosen = [r for r in rows if r['domain'] == domain and r['split'] == split]
    model = models.get(domain, models.get('combined'))
    if not chosen:
        raise ValueError(f'Empty development partition: {domain}/{split}')
    indices = [int(np.argmax(model.predict(r['features']))) for r in chosen]
    return {'N': len(chosen), 'sari_final': 100 * float(np.mean([r['sari'][i] for r, i in zip(chosen, indices)])),
            'exact': float(np.mean([r['exact'][i] for r, i in zip(chosen, indices)])),
            'changed_rate': float(np.mean([r['changed'][i] for r, i in zip(chosen, indices)]))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, required=True, help='Development features produced by build_utility_features.py.')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--search', action='store_true')
    a = p.parse_args()
    rows = read_jsonl(a.features)
    if any(r['split'] not in ('train', 'guard', 'holdout') or r['domain'] not in ('fecdata', 'scifact') for r in rows):
        raise ValueError('Utility fitting only accepts declared development partitions')
    selected = {'mode': 'per_domain', 'exact_weight': 1.0, 'depth': 2}
    search = []
    if a.search:
        for mode in ('combined', 'per_domain'):
            for weight in (0.0, 0.25, 0.5, 1.0):
                for depth in (2, 3):
                    models = fit(rows, mode, weight, depth, False)
                    guard = {d: report(rows, models, 'guard', d) for d in ('fecdata', 'scifact')}
                    values = [x['sari_final'] + 100 * x['exact'] for x in guard.values()]
                    search.append({'mode': mode, 'exact_weight': weight, 'depth': depth, 'guard': guard,
                                   'objective': min(values) + 0.25 * float(np.mean(values))})
        search.sort(key=lambda r: (r['objective'], r['guard']['scifact']['exact'], r['guard']['fecdata']['exact']), reverse=True)
        selected = search[0]
    models = fit(rows, selected['mode'], selected['exact_weight'], selected['depth'], True)
    write_json(a.output, export_utility(models, selected))
    write_json(a.output.with_suffix('.report.json'), {'selected': selected, 'search': search,
        'features_sha256': sha256(a.features),
        'holdout': {d: report(rows, models, 'holdout', d) for d in ('fecdata', 'scifact')}})
    print(f'Refitted utility model: {a.output}')


if __name__ == '__main__':
    main()
