# Data

| Use | Source and download instructions |
| --- | --- |
| FECDATA / FEVER-FEC | [Original FECDATA repository](https://github.com/j6mes/acl2021-factual-error-correction); the provided-evidence files used by our scripts are available in [LIFE's processed data folder](https://drive.google.com/drive/folders/16H7WA8rZ-qlGNNIgylrPQox6qAdQq_qM?usp=sharing). |
| SciFact-FEC | Download the correction dataset through [ZeroFEC's data instructions](https://github.com/khuangaf/ZeroFEC#data), including `scifact_test.jsonl`. |
| Perturbation pairs for planner training | Follow [LIFE's synthetic-data preparation](https://github.com/NLPCode/LIFE#train-our-proposed-life-for-factual-error-correction) to obtain filtered pairs and the gold-evidence factuality verifier. |

Keep downloaded data under `data/` and set the training input paths in `configs/paths.local.toml`. The training scripts turn perturbation pairs into candidate-level locality supervision. Dataset processing and experimental settings follow the paper.
