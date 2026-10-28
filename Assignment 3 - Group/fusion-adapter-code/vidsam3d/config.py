"""Config loading: base <- condition <- env <- CLI overrides (OmegaConf)."""

from __future__ import annotations

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_ENV = "local"


def _load_yaml(path: Path) -> DictConfig:
    """Load one YAML file, failing with a clear message if it does not exist."""
    if not path.is_file():
        options = sorted(p.stem for p in path.parent.glob("*.yaml"))
        raise FileNotFoundError(f"{path} not found; available: {options}")
    return OmegaConf.load(path)


def load_config(
    condition: str | None = None,
    env: str | None = None,
    overrides: list[str] | None = None,
    config_dir: Path = CONFIG_DIR,
) -> DictConfig:
    """Merge base <- conditions/<condition> <- env/<env> <- `key=value` overrides.

    The env defaults to $VIDSAM3D_ENV, then "local". The base config is in struct mode,
    so condition/env files and overrides can only change keys that exist in base.yaml.
    """
    env = env or os.environ.get("VIDSAM3D_ENV") or DEFAULT_ENV
    cfg = _load_yaml(config_dir / "base.yaml")
    OmegaConf.set_struct(cfg, True)
    layers = []
    if condition:
        layers.append(_load_yaml(config_dir / "conditions" / f"{condition}.yaml"))
    layers.append(_load_yaml(config_dir / "env" / f"{env}.yaml"))
    layers.append(OmegaConf.from_dotlist(list(overrides or [])))
    cfg = OmegaConf.merge(cfg, *layers)
    cfg.condition = condition or "base"
    cfg.env = env
    return cfg


def save_config(cfg: DictConfig, run_dir: str | Path) -> Path:
    """Write the resolved config to `<run_dir>/config.yaml` and return the path."""
    path = Path(run_dir) / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    return path


def apply_process_env(cfg: DictConfig) -> None:
    """Export settings that libraries read from the environment (call before importing them)."""
    os.environ.setdefault("HF_HOME", str(cfg.paths.hf_home))
