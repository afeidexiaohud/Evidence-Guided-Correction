"""Acquire the external backend at the pinned commit and apply the project scorer patch."""
import argparse
import json
import subprocess
from pathlib import Path
from egip.backend import validate_backend


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dest', type=Path, default=Path('third_party/MaskToCorrect'))
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    pin = json.loads((root / 'resources/backend-v4.json').read_text())
    dest = a.dest.resolve()
    if dest.exists():
        validate_backend(dest)
        print(f'Existing backend passes v4 hash checks: {dest}')
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', '--no-checkout', pin['repository'], str(dest)], check=True)
    subprocess.run(['git', '-C', str(dest), 'checkout', '--detach', pin['commit']], check=True)
    subprocess.run(['git', '-C', str(dest), 'apply', str(root / 'patches/m2c-scorer-v4.patch')], check=True)
    validate_backend(dest)
    print(f'Prepared v4 shared corrector backend: {dest}')


if __name__ == '__main__':
    main()
