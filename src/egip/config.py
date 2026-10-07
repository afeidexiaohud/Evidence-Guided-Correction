from __future__ import annotations
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def load_config(path: str | Path) -> dict:
    """Load typed TOML; relative filesystem paths are relative to project.root.

    project.root is relative to the config directory. Hugging Face identifiers
    remain identifiers; local model paths should start with ./ or /.
    """
    path = Path(path).expanduser().resolve()
    with path.open('rb') as handle:
        config = tomllib.load(handle)
    root = Path(config['project']['root']).expanduser()
    if not root.is_absolute():
        root = path.parent / root
    root = root.resolve()
    config['project']['root'] = str(root)
    for section in ('data', 'outputs', 'backends'):
        for key, value in config.get(section, {}).items():
            value_path = Path(value).expanduser()
            config[section][key] = str(value_path if value_path.is_absolute() else root / value_path)
    for key, value in config.get('models', {}).items():
        if isinstance(value, str) and (value.startswith(('.', '/', '~')) or key.startswith('life_verifier')):
            value_path = Path(value).expanduser()
            config['models'][key] = str(value_path if value_path.is_absolute() else root / value_path)
    return config
