# Running EGIP

These commands implement the v4 provided-evidence setting. Run them from the repository root after following [installation](INSTALL.md) and downloading the [data](DATA.md).

## Inference and evaluation

The example below uses FECDATA. For SciFact-FEC, replace `fecdata` with `scifact` in paths and `--domain` arguments.

### Prepare the data and intervention plan

```bash
python scripts/prepare_data.py --domain fecdata \
  --source data/upstream/fecdata/test.jsonl --output data/fecdata.canonical.jsonl
python scripts/prepare_benchmark_candidates.py \
  --canonical data/fecdata.canonical.jsonl --output artifacts/fecdata.candidates.jsonl
python scripts/score_benchmark_candidates.py \
  --input artifacts/fecdata.candidates.jsonl \
  --checkpoint models/priority-v4/checkpoint.pt \
  --backbone models/priority-v4/backbone --tokenizer models/priority-v4/tokenizer \
  --device cuda --precision bfloat16 --batch-size 64 \
  --output artifacts/fecdata.scored.jsonl
python scripts/rank_benchmark_candidates.py \
  --input artifacts/fecdata.scored.jsonl --output artifacts/fecdata.ranked.jsonl
```

For SciFact-FEC, the upstream source file is `data/upstream/scifact_test.jsonl`. The priority checkpoint download is pending; the inference example requires that checkpoint or a model trained using the procedure below.

Candidate phrases span multiple lexical granularities. The planner uses priority weight 0.9 and diversity weight 0.3, retaining overlapping candidates. This corresponds to the paper's intervention-space construction and plan allocation.

### Generate candidate-conditioned revisions

```bash
python scripts/run_correction.py \
  --ranked artifacts/fecdata.ranked.jsonl --domain fecdata --k 2 \
  --backend third_party/MaskToCorrect \
  --model /path/to/Qwen2.5-32B-Instruct \
  --docnli /path/to/DocNLI.pretrained.RoBERTA.model.pt \
  --roberta /path/to/roberta-large \
  --scorer-python /path/to/scorer-environment/bin/python \
  --work-dir outputs/fecdata-correction --output artifacts/fecdata.bank.jsonl
```

Each selected surface form is replaced with `[MASK]` in the original claim. Correction attempts are independent; the unchanged input is added separately. The shared backend uses Qwen2.5-32B-Instruct with the generation settings in the paper. It generates both EGIP and MMR candidates once so the comparison can share the same downstream inputs.

### Select the final revision and evaluate

```bash
python scripts/select_revision.py \
  --bank artifacts/fecdata.bank.jsonl --ranked artifacts/fecdata.ranked.jsonl \
  --domain fecdata --k 2 --planner egip --selector full \
  --output outputs/fecdata.predictions.jsonl
python scripts/evaluate.py \
  --canonical data/fecdata.canonical.jsonl --predictions outputs/fecdata.predictions.jsonl \
  --expected-n 3882 \
  --output outputs/fecdata.metrics.json
```

For SciFact-FEC, use `--expected-n 100`. The full selector uses the included per-domain utility models and context router. `--planner mmr` and `--selector base` select the comparison configurations in the paper. Evaluation reports SARI and Exact Correction (%). Test references are used for evaluation, not planning or revision selection.

## Training the priority model

Set the filtered perturbation-pair, benchmark and factuality-verifier paths in `configs/paths.local.toml`, then run:

```bash
python scripts/export_life_traces.py --setting gold --source filtered
for split in train dev heldout; do
  python scripts/score_trace_factuality.py --setting gold --source filtered --split "$split"
  python scripts/build_locator_dataset.py --setting gold --source filtered --split "$split"
done
python scripts/audit_leakage.py --setting gold --source filtered
python scripts/train_locator.py --setting gold_filtered --device cuda \
  --output-dir outputs/retrained-priority
```

The default training parameters follow the paper: NLI RoBERTa backbone, locality IoU threshold 0.5, weighted binary cross entropy, two epochs, learning rate 2e-5, batch size 16, gradient accumulation 2, maximum length 512 and seed 13. The checkpoint is selected by development candidate-ranking MRR.

## Training the utility selector

To fit a new utility model, prepare development correction banks, ranked candidates, references and split assignments. `build_utility_features.py` extracts revision features and targets; `train_utility.py` fits the paper's per-domain regressors:

```bash
python scripts/build_utility_features.py \
  --bank data/dev.bank.jsonl --ranked data/dev.ranked.jsonl \
  --references data/dev.references.jsonl --splits data/dev.splits.jsonl \
  --domain fecdata --output data/fecdata.utility-features.jsonl
# Repeat for SciFact-FEC, then combine the two feature files.
cat data/fecdata.utility-features.jsonl data/scifact.utility-features.jsonl > data/utility-features.jsonl
python scripts/train_utility.py --features data/utility-features.jsonl \
  --output outputs/utility.json
```

Development banks use K=5. Reference JSONL contains `record_id, reference`; split JSONL contains `record_id, split`, with `train`, `guard` and `holdout` partitions. Training uses development data only. The fitted models used in the paper are included for inference.

## Additional paper analyses

`scripts/replay_budget.py` runs the K=1–10 comparison from a top-10 correction bank. `scripts/analyze_interventions.py` computes intervention overlap and duplicate-revision statistics; `scripts/evaluate_human.py` evaluates supplied locality annotations. See each script's `--help` for arguments. Experimental findings are reported in the paper.
