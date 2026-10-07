"""Acquire a release artifact by explicit URL or file and enforce its frozen hash."""
import argparse
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path
from egip.io import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', choices=('priority',), default='priority')
    p.add_argument('--source', required=True, help='Artifact URL or local archive path; private assets can be downloaded with gh first.')
    p.add_argument('--output-dir', type=Path, default=Path('models'))
    a = p.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'resources/artifacts-v4.json').read_text())
    expected = manifest[a.name]['archive_sha256']
    a.output_dir.mkdir(parents=True, exist_ok=True)
    archive = a.output_dir / 'egip-priority-v4.tar.gz'
    temporary = archive.with_suffix('.download')
    if a.source.startswith(('https://', 'http://')):
        with urllib.request.urlopen(a.source, timeout=60) as response, temporary.open('wb') as target:
            shutil.copyfileobj(response, target)
    else:
        shutil.copyfile(a.source, temporary)
    if sha256(temporary) != expected:
        temporary.unlink()
        raise ValueError('Artifact SHA-256 mismatch')
    temporary.replace(archive)
    with tarfile.open(archive) as tar:
        base = a.output_dir.resolve()
        for member in tar.getmembers():
            path = (base / member.name).resolve()
            if not path.is_relative_to(base) or not (member.isdir() or member.isfile()):
                raise ValueError('Unsafe archive member')
        for member in tar.getmembers():
            path = a.output_dir / member.name
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, path.open('wb') as target:
                    shutil.copyfileobj(source, target)
    print(f'Verified and extracted priority-v4 to {a.output_dir}')


if __name__ == '__main__':
    main()
