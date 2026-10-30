"""Read-only checks for the shared environment and installed model dependencies."""
from __future__ import annotations
import importlib.util
import importlib.metadata
from pathlib import Path
import chz
import torch
from video3d.config import ExperimentConfig

Row = tuple[str, str, str]

def torch_supports(capability: tuple[int, int], arch_list: list[str]) -> bool:
    """True if a torch build with `arch_list` (e.g. ["sm_80", "compute_90"]) can run on the GPU.

    SASS runs on GPUs of the same major version with an equal or higher minor version;
    PTX (compute_XY) can be JIT-compiled for any newer GPU.
    """
    major, minor = capability
    for arch in arch_list:
        kind, num = arch.split("_")
        m, n = divmod(int(num), 10)
        if kind == "sm" and m == major and n <= minor:
            return True
        if kind == "compute" and (m, n) <= (major, minor):
            return True
    return False

def check_env(cfg: ExperimentConfig, network: bool = True, load_models: bool = False) -> list[Row]:
    rows: list[Row] = []
    for module in ["torch", "numpy", "chz", "sam3d_objects", "src", "pytorch3d", "kaolin"]:
        present = importlib.util.find_spec(module) is not None
        rows.append(("OK" if present else "WARN", module, "installed" if present else "not installed"))
    for key, value in chz.asdict(cfg.paths).items():
        path = Path(value).expanduser()
        rows.append(("OK" if path.exists() else "WARN", f"paths.{key}", str(path)))
    rows.append(("OK" if torch.cuda.is_available() else "WARN", "cuda", str(torch.cuda.is_available())))
    if network:
        from huggingface_hub import auth_check
        auth_check(cfg.sam3d.hf_repo)
    if load_models:
        from video3d.models.sam3d import build_sam3d
        from video3d.models.encoders import build_kv_encoder
        device = torch.device(cfg.device if cfg.device != "auto" else "cuda")
        sam = build_sam3d(cfg, device)
        build_kv_encoder(cfg, sam, device)
        rows.append(("OK", "models", "loaded"))
    for status, name, detail in rows:
        print(f"[{status}] {name}: {detail}")
    print(f"{sum(status == 'FAIL' for status, _, _ in rows)} failed")
    return rows
