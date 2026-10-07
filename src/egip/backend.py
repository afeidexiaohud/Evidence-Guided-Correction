"""Explicit integration with the pinned external shared corrector."""
from pathlib import Path
import json
from .io import sha256


def validate_backend(path):
    root = Path(__file__).resolve().parents[2]
    pin = json.loads((root / 'resources/backend-v4.json').read_text())
    path = Path(path).resolve()
    for name, expected in pin['runtime_files'].items():
        if sha256(path / name) != expected:
            raise ValueError(f'Shared backend hash differs from v4: {name}')
    return path
