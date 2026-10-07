# Contributing

Contributions to EGIP are welcome. Describe the change and how it affects the method or usage. When changing algorithm behavior, identify the relevant paper component and include a focused test.

Run the existing tests before submitting:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

Use development data for model fitting and parameter selection; keep test references out of inference. Preserve third-party notices and keep data, model weights, credentials and local experiment outputs outside Git.
