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
