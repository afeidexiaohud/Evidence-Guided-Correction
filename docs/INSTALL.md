# Installation

Use Python 3.10 or later. Install the priority model and ranking dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train]'
cp configs/paths.example.toml configs/paths.local.toml
```

## Models

| Role | Resource |
| --- | --- |
| EGIP priority model | Checkpoint download pending; expected directory: `models/priority-v4/`. |
| Priority-training backbone | [cross-encoder/nli-roberta-base](https://huggingface.co/cross-encoder/nli-roberta-base) |
| Relevance and diversity | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Shared corrector | [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) |
| Correction confidence | [DocNLI pretrained checkpoint](https://github.com/salesforce/DocNLI#pretrained-roberta-model-download), with [RoBERTa-large](https://huggingface.co/FacebookAI/roberta-large) |
| Downstream utility | Included in `resources/utility-v4.json`. |

Download the external models from their authors. Model versions used by the implementation are recorded in `resources/artifacts-v4.json`. Configure local training paths in `configs/paths.local.toml`; inference commands accept model paths directly.

## Correction backend

```bash
python -m pip install -e '.[generate]'
python scripts/prepare_backend.py --dest third_party/MaskToCorrect
```

The installer obtains the paper's shared [MaskToCorrect backend](https://github.com/payelsantra/MaskToCorrect) and applies the scorer compatibility patch. Install the scorer dependencies in a separate environment if needed:

```bash
python -m pip install -e '.[score]'
```

Qwen generation uses two GPUs with vLLM, as in the paper. Pass the scorer environment's Python executable to `run_correction.py` when using separate environments. CPU installation and tests do not establish GPU generation reproducibility.
