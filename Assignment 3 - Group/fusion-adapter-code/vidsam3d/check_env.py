"""`check-env`: report GPU, library versions, paths and Hugging Face access. Never raises."""

from __future__ import annotations

import importlib
import os
import platform
import shutil
from pathlib import Path

from omegaconf import DictConfig

MIN_VRAM_GB = 32
PACKAGES = ["torch", "numpy", "PIL", "omegaconf", "scipy", "matplotlib", "huggingface_hub",
            "pytorch3d", "kaolin"]

Row = tuple[str, str, str]  # (status, name, detail); status in OK | WARN | FAIL | SKIP


def check_packages() -> list[Row]:
    """Import each package and report its version."""
    rows = []
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            rows.append(("OK", name, getattr(mod, "__version__", "?")))
        except Exception as e:  # missing or broken install
            rows.append(("WARN", name, f"not importable ({type(e).__name__})"))
    return rows


def check_gpu() -> list[Row]:
    """Report CUDA availability, devices, VRAM and bf16 support."""
    try:
        import torch
    except ImportError:
        return [("FAIL", "torch", "not installed")]
    if not torch.cuda.is_available():
        return [("WARN", "cuda", f"not available (torch {torch.__version__}); only CPU tests will run")]
    rows = [("OK", "cuda", f"CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}")]
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gb = props.total_memory / 1024**3
        status = "OK" if gb >= MIN_VRAM_GB else "WARN"
        rows.append((status, f"gpu{i}", f"{props.name}, {gb:.1f} GB (SAM 3D needs >= {MIN_VRAM_GB})"))
    rows.append(("OK" if torch.cuda.is_bf16_supported() else "WARN", "bf16",
                 str(torch.cuda.is_bf16_supported())))
    return rows


def check_paths(cfg: DictConfig) -> list[Row]:
    """Check that every configured path exists; report free space on scratch."""
    rows = []
    for key in cfg.paths:
        path = Path(str(cfg.paths[key]))
        rows.append(("OK" if path.exists() else "WARN", f"paths.{key}", str(path)))
    scratch = Path(str(cfg.paths.scratch_root))
    probe = scratch if scratch.exists() else scratch.resolve().parent
    if probe.exists():
        free_gb = shutil.disk_usage(probe).free / 1024**3
        rows.append(("OK", "scratch free", f"{free_gb:.0f} GB at {probe}"))
    return rows


def check_hf(cfg: DictConfig, network: bool) -> list[Row]:
    """Check that an HF token is set and grants access to the gated SAM 3D repo."""
    try:
        import huggingface_hub as hf
    except ImportError:
        return [("FAIL", "huggingface_hub", "not installed")]
    rows = [("OK", "HF_HOME", os.environ.get("HF_HOME", "(unset)"))]
    token = hf.get_token()  # reads HF_TOKEN or the cached login; never printed
    rows.append(("OK" if token else "WARN", "HF token", "found" if token else "missing (set HF_TOKEN)"))
    repo = str(cfg.sam3d.hf_repo)
    if not network:
        rows.append(("SKIP", f"HF access {repo}", "--offline"))
        return rows
    try:
        hf.auth_check(repo)
        rows.append(("OK", f"HF access {repo}", "granted"))
    except Exception as e:
        rows.append(("FAIL", f"HF access {repo}", f"{type(e).__name__}: {str(e).splitlines()[0]}"))
    return rows


def check_models(cfg: DictConfig) -> list[Row]:
    """SAM 3D import and V-JEPA loadability (implemented in P5)."""
    return [("SKIP", "SAM 3D import", "implemented in P5"),
            ("SKIP", "V-JEPA 2.1 load", "implemented in P5")]


def check_env(cfg: DictConfig, network: bool = True) -> list[Row]:
    """Run all checks and print a table; return the rows."""
    rows: list[Row] = [("OK", "python", f"{platform.python_version()} on {platform.platform()}"),
                       ("OK", "env", f"{cfg.env} (condition {cfg.condition})")]
    for check in (check_packages, check_gpu):
        rows += check()
    rows += check_paths(cfg)
    rows += check_hf(cfg, network)
    rows += check_models(cfg)
    width = max(len(name) for _, name, _ in rows)
    for status, name, detail in rows:
        print(f"[{status:4s}] {name:<{width}}  {detail}")
    n_bad = sum(status == "FAIL" for status, _, _ in rows)
    n_warn = sum(status == "WARN" for status, _, _ in rows)
    print(f"\n{n_bad} failed, {n_warn} warnings")
    return rows
