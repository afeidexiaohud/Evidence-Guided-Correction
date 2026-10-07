"""Frozen v4 reference-free utility features and portable tree inference."""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np

FEATURE_NAMES = [
    "correction_confidence", "source_rouge", "confidence_margin_vs_input",
    "prediction_evidence_overlap", "overlap_delta_vs_input", "edit_ratio",
    "token_length_ratio", "absolute_length_ratio_delta", "no_edit",
    "input_fallback", "planner_position", "locator_error_probability",
    "semantic_relevance", "candidate_width_ratio", "claim_token_length",
    "evidence_token_length", "prediction_has_negation", "source_has_negation",
    "domain_scifact",
]

def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))

def overlap(left: str, right: str) -> float:
    a, b = tokens(left), tokens(right)
    return 2.0 * len(a & b) / (len(a) + len(b)) if a and b else 0.0

def edit_ratio(source: str, target: str) -> float:
    a, b = tokens(source), tokens(target)
    return 1.0 - len(a & b) / max(1, len(a | b))

def make_features(item: dict, output_index: int, position: int, candidate: dict | None, domain: str) -> list[float]:
    source = str(item["input_claim"])
    prediction = str(item["corrected sentences"][output_index]).strip()
    evidence = " ".join(item["evidence"])
    source_overlap = overlap(source, evidence)
    prediction_overlap = overlap(prediction, evidence)
    input_index = len(item["corrected sentences"]) - 1
    confidence = float(item["correction_scores"][output_index])
    input_confidence = float(item["correction_scores"][input_index])
    source_tokens, prediction_tokens = tokens(source), tokens(prediction)
    length_ratio = len(prediction_tokens) / max(1, len(source_tokens))
    has_negation = lambda values: float(bool(values & {"not", "no", "never", "none", "without"}))
    return [
        confidence,
        float(item["rouge_scores"][output_index]),
        confidence - input_confidence,
        prediction_overlap,
        prediction_overlap - source_overlap,
        edit_ratio(source, prediction),
        length_ratio,
        abs(length_ratio - 1.0),
        float(prediction.strip() == source.strip()),
        float(candidate is None),
        position / 5.0,
        float(candidate["error_probability"]) if candidate is not None else 0.0,
        float(candidate["relevance"]) if candidate is not None else source_overlap,
        ((int(candidate["end"]) - int(candidate["start"])) / max(1, len(source))) if candidate is not None else 0.0,
        len(source_tokens) / 50.0,
        len(tokens(evidence)) / 200.0,
        has_negation(prediction_tokens),
        has_negation(source_tokens),
        float(domain == "scifact"),
    ]


class UtilityModel:
    """Evaluate exported frozen histogram trees on raw features, without pickle.

    Node values already include the fitted learning rate. Preserve tree order,
    <= thresholds and missing-value branches from scikit-learn 1.2.2.
    """
    def __init__(self, path: str | Path):
        self.artifact = json.loads(Path(path).read_text())
        if self.artifact['format'] != 'egip-hist-trees-v1':
            raise ValueError('Unsupported utility model format')
        if self.artifact['feature_names'] != FEATURE_NAMES:
            raise ValueError('Utility feature order mismatch')

    def predict(self, features, domain: str):
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
            raise ValueError('Expected a matrix with 19 ordered utility features')
        model = self.artifact['models'][domain]
        result = np.full(len(x), model['baseline'], dtype=np.float64)
        for tree in model['trees']:
            pending = [(0, np.arange(len(x)))]
            while pending:
                index, rows = pending.pop()
                if not len(rows):
                    continue
                node = tree[index]
                if node['is_leaf']:
                    result[rows] += node['value']
                    continue
                if node['is_categorical']:
                    raise ValueError('Categorical tree nodes are not supported')
                values = x[rows, node['feature_idx']]
                left = np.where(np.isnan(values), bool(node['missing_go_to_left']), values <= node['num_threshold'])
                pending.extend([(node['right'], rows[~left]), (node['left'], rows[left])])
        return result


def export_utility(models: dict, config: dict) -> dict:
    """Export retrained sklearn models to the same portable representation."""
    output = {'format': 'egip-hist-trees-v1', 'feature_names': FEATURE_NAMES,
              'config': config, 'models': {}}
    for domain, model in models.items():
        trees = []
        for predictors in model._predictors:
            if len(predictors) != 1 or predictors[0].nodes['is_categorical'].any():
                raise ValueError('Expected one noncategorical regression tree per iteration')
            trees.append([{name: (int(value) if value.dtype.kind in 'ui' else float(value))
                           for name, value in zip(predictors[0].nodes.dtype.names, node)}
                          for node in predictors[0].nodes])
        output['models'][domain] = {'baseline': float(model._baseline_prediction[0, 0]), 'trees': trees}
    return output
