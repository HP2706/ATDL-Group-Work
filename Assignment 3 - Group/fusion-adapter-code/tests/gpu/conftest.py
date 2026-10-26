"""Fixtures for GPU tests: the real config, SAM 3D and V-JEPA (loaded once per session).

Run on a GPU machine with `VIDSAM3D_ENV=<env> pytest -m gpu tests/gpu`. Tests skip when CUDA,
the SAM 3D checkpoints or the eval data are missing.
"""

from pathlib import Path

import pytest
import torch

from vidsam3d.config import load_config


@pytest.fixture(scope="session")
def gpu_cfg():
    """Config from $VIDSAM3D_ENV; skips if checkpoints or eval data are missing."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    cfg = load_config("c3")
    if not (Path(cfg.paths.sam3d_checkpoint_dir) / "pipeline.yaml").is_file():
        pytest.skip(f"no SAM 3D checkpoints in {cfg.paths.sam3d_checkpoint_dir}")
    if not (Path(cfg.paths.data_root) / "manifest.jsonl").is_file():
        pytest.skip(f"no dataset at {cfg.paths.data_root}")
    return cfg


@pytest.fixture(scope="session")
def sam3d(gpu_cfg):
    from vidsam3d.models.sam3d import Sam3DGeometry
    return Sam3DGeometry(gpu_cfg, torch.device("cuda"))


@pytest.fixture(scope="session")
def vjepa(gpu_cfg):
    from vidsam3d.models.encoders import VJEPAEncoder
    return VJEPAEncoder(gpu_cfg, torch.device("cuda"))


@pytest.fixture(scope="session")
def eval_sample(gpu_cfg):
    from vidsam3d.data.samples import ClipDataset
    return ClipDataset(gpu_cfg, gpu_cfg.data.eval_split, train=False)[0]


@pytest.fixture(scope="session")
def cond(sam3d, eval_sample):
    x = eval_sample["inputs"]
    return sam3d.condition(x["ref_rgb"][None].cuda(), x["ref_mask"][None].cuda())
