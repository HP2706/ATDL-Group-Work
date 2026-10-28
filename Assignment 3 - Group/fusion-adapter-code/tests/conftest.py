"""Shared pytest setup: skip `@pytest.mark.gpu` tests without CUDA."""

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip GPU-marked tests when CUDA is unavailable."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if has_cuda:
        return
    skip = pytest.mark.skip(reason="needs CUDA (and SAM 3D)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def fake_root(tmp_path_factory):
    """Root of fake training exports: <root>/train (4 clips), <root>/eval (2 clips), 64x64."""
    from tests.fixtures.make_fake_dataset import make_fake_dataset
    return make_fake_dataset(tmp_path_factory.mktemp("dataset"))


@pytest.fixture
def fake_cfg(fake_root, tmp_path):
    """Local config pointing at the fake dataset, with a crop size that fits 64x64 frames."""
    from vidsam3d.config import load_config
    return load_config("c3", "local", [f"paths.data_root={fake_root}", f"paths.cache_root={tmp_path}/cache",
                                       "data.crop.min_side=16", "data.crop.size=32"])
