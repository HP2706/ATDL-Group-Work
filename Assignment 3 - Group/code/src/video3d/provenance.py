"""Source and data fingerprints used to reject incompatible resumes."""
from __future__ import annotations
import hashlib
from pathlib import Path
from video3d.config import ExperimentConfig


def source_identity() -> str:
    root=Path(__file__).parent
    digest=hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def data_identity(cfg: ExperimentConfig) -> str:
    if cfg.data.source in {"synthetic", "omniobject3d", "gso"}:
        from video3d.data.targets import cache_identity
        return cache_identity(cfg)
    return hashlib.sha256((Path(cfg.data.prepared)/"manifest.jsonl").read_bytes()).hexdigest()
