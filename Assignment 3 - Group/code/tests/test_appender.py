"""Alignment, padding and gradient contracts for separate appended streams."""
from __future__ import annotations

import pytest
import torch

from video3d.config import AdapterConfig
from video3d.models.adapter import build_adapter
from video3d.models.conditioning import VideoFeatures
from video3d.models.encoders import mask_patch_descriptors


def features() -> VideoFeatures:
    return VideoFeatures(tokens=torch.randn(2, 6, 32, requires_grad=True),
                         coverage=torch.ones(2, 6), dt=torch.zeros(2, 6, dtype=torch.long),
                         padding=torch.tensor([[False] * 4 + [True] * 2, [False] * 6]),
                         mask_patches=torch.rand(2, 6, 32), positions=torch.randn(2, 6, 3))


def test_appender_preserves_native_and_excludes_padding_with_gradients() -> None:
    model = build_adapter(AdapterConfig(strategy="append_video_mask", d_vid=32,
                                        d_cond=16, heads=4, append_queries=5))
    video = features()
    native = {"image": torch.randn(2, 4, 16), "mask": torch.randn(2, 4, 16)}
    output = model.condition(native, video, [])
    assert list(output) == ["image", "mask", "video_rgb", "video_mask"]
    assert output["image"] is native["image"] and output["mask"] is native["mask"]
    assert output["video_rgb"].shape == output["video_mask"].shape == (2, 5, 16)
    changed = video.model_copy(update={"tokens": video.tokens.detach().clone(),
                                       "mask_patches": video.mask_patches.clone(),
                                       "positions": video.positions.clone()})
    changed.tokens[0, 4:] = 1000
    changed.mask_patches[0, 4:] = 1000
    changed.positions[0, 4:] = 1000
    second = model.condition(native, changed, [])
    for name in ["video_rgb", "video_mask"]:
        torch.testing.assert_close(output[name], second[name])
    loss = output["video_rgb"].square().mean() + output["video_mask"].square().mean()
    loss.backward()
    assert torch.isfinite(video.tokens.grad).all()
    assert video.tokens.grad[0, 4:].abs().sum() == 0
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert model.mask_projection[0].weight.grad.abs().sum() > 0
    assert model.condition(native, video, [], video_disabled=True) is native


def test_mask_descriptors_keep_tubelet_order_and_silhouette() -> None:
    masks = torch.zeros(4, 16, 32)
    masks[0, :, :8] = 1
    masks[1, :, 8:16] = 1
    masks[2:, :, 16:] = 1
    descriptor = mask_patch_descriptors(masks, 16, 2)
    assert descriptor.shape == (4, 32)
    assert descriptor[0, :16].reshape(4, 4)[:, :2].eq(1).all()
    assert descriptor[0, 16:].reshape(4, 4)[:, 2:].eq(1).all()
    assert descriptor[1].eq(0).all() and descriptor[2].eq(0).all()
    assert descriptor[3].eq(1).all()


def test_invisible_frames_and_bfloat16_are_finite_but_all_padding_fails() -> None:
    model = build_adapter(AdapterConfig(strategy="append_video_mask", d_vid=32,
                                        d_cond=16, heads=4, append_queries=5))
    video = features()
    video.coverage.zero_()
    video.mask_patches.zero_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model.condition({"image": torch.randn(2, 4, 16)}, video, [])
        (output["video_rgb"].square().mean() + output["video_mask"].square().mean()).backward()
    assert all(torch.isfinite(tensor).all() for tensor in output.values())
    video.padding.fill_(True)
    with pytest.raises(ValueError, match="no valid"):
        model.condition({"image": torch.randn(2, 4, 16)}, video, [])
