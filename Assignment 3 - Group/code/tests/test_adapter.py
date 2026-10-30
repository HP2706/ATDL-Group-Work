"""Token coverage and the gated fusion adapter (identity at init, gradients, padding)."""
import typing

import pytest
import torch
import torch.nn as nn

from video3d.config import load_config
from video3d.data.samples import ClipDataset
from video3d.models.adapter import (VideoFusionAdapter, attention_mass_on_object, build_adapter,
                                     count_parameters, pad_video_batch)
from video3d.models.encoders import per_token, token_coverage

D = 64  # small width for fast CPU tests


def small_adapter(**kw: typing.Any) -> typing.Any:
    return VideoFusionAdapter(d_cond=D, d_vid=D, heads=4, **kw)


def inputs(B: typing.Any=2, Nc: typing.Any=5, N: typing.Any=12, seed: typing.Any=0) -> typing.Any:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, Nc, D, generator=g), torch.randn(B, N, D, generator=g),
            torch.rand(B, N, generator=g), torch.randint(-3, 4, (B, N), generator=g))


def open_gates(adapter: typing.Any, value: typing.Any=0.5) -> typing.Any:
    with torch.no_grad():
        for blk in adapter.blocks:
            blk.g_attn.fill_(value)
            blk.g_ff.fill_(value)


# ---- coverage ----

@pytest.mark.parametrize("patch, tubelet, size", [(16, 2, 32), (14, 2, 28), (14, 1, 28)])
def test_coverage_preserves_mask_area(patch: typing.Any, tubelet: typing.Any, size: typing.Any) -> typing.Any:
    mask = torch.rand(2, 4, size, size)
    cov = token_coverage(mask, patch, tubelet)
    T = 4 // tubelet
    assert cov.shape == (2, T * (size // patch) ** 2)
    area = cov.sum(1) * patch**2 * (4 // T)
    torch.testing.assert_close(area, mask.sum((1, 2, 3)))


def test_coverage_on_real_sample_crops(fake_cfg: typing.Any) -> typing.Any:
    frame_masks = ClipDataset(fake_cfg, "eval", train=False)[0]["inputs"]["frame_masks"][None]
    cov = token_coverage(frame_masks, patch=16, tubelet=2)
    torch.testing.assert_close(cov.sum() * 16**2 * 2, frame_masks.sum())
    assert 0 <= cov.min() and cov.max() <= 1


@pytest.mark.parametrize("t, h, w", [(0, 0, 0), (1, 0, 1), (3, 1, 0), (2, 1, 1)])
def test_coverage_token_order_is_t_h_w(t: typing.Any, h: typing.Any, w: typing.Any) -> typing.Any:
    mask = torch.zeros(1, 8, 32, 32)
    mask[0, 2 * t:2 * t + 2, 16 * h:16 * h + 16, 16 * w:16 * w + 16] = 1
    cov = token_coverage(mask, 16, 2)[0]
    assert cov.nonzero().flatten().tolist() == [t * 4 + h * 2 + w]


def test_coverage_single_frame_and_odd_k() -> typing.Any:
    assert token_coverage(torch.ones(1, 1, 32, 32), 16, 2).shape == (1, 4)   # image tokenizer
    with pytest.raises(AssertionError):
        token_coverage(torch.ones(1, 3, 32, 32), 16, 2)


def test_per_token() -> typing.Any:
    assert per_token(torch.tensor([[-1, 0, 1]]), 2).tolist() == [[-1, -1, 0, 0, 1, 1]]


# ---- adapter ----

def test_parameter_count() -> typing.Any:
    adapter = build_adapter(load_config("c3").adapter)
    assert count_parameters(adapter) == 17_869_828                       # ≈ 17.9M
    nomask = build_adapter(load_config("c3_nomask").adapter)
    assert nomask.e_obj is None and count_parameters(nomask) == 17_869_828 - 1024


def test_identity_at_init() -> typing.Any:
    adapter = small_adapter()
    c, f, m, dt = inputs()
    assert torch.equal(adapter(c, f, m, dt), c)
    assert all(g == 0 for g in adapter.gates().values())


def test_only_gates_get_gradient_at_step_0() -> typing.Any:
    adapter = small_adapter()
    c, f, m, dt = inputs()
    (adapter(c, f, m, dt) * torch.randn_like(c)).sum().backward()
    for name, p in adapter.named_parameters():
        if name.split(".")[-1] in ("g_attn", "g_ff"):
            assert p.grad.abs().sum() > 0, name
        else:
            assert p.grad is None or p.grad.abs().sum() == 0, name


def test_all_weights_train_after_a_few_steps() -> typing.Any:
    adapter = small_adapter()
    opt = torch.optim.SGD(adapter.parameters(), lr=0.1)
    c, f, m, dt = inputs()
    target = torch.randn_like(c)
    for _ in range(3):
        opt.zero_grad()
        ((adapter(c, f, m, dt) - target) ** 2).mean().backward()
        opt.step()
    for name, p in adapter.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_frozen_modules_get_no_gradient() -> typing.Any:
    frozen_cond, frozen_enc = nn.Linear(D, D), nn.Linear(D, D)
    for p in list(frozen_cond.parameters()) + list(frozen_enc.parameters()):
        p.requires_grad_(False)
    adapter = small_adapter()
    open_gates(adapter)
    c, f, m, dt = inputs()
    with torch.no_grad():
        c, f = frozen_cond(c), frozen_enc(f)
    adapter(c, f, m, dt).sum().backward()
    assert all(p.grad is None for p in list(frozen_cond.parameters()) + list(frozen_enc.parameters()))
    assert adapter.in_proj.weight.grad.abs().sum() > 0


def test_padding_lets_batches_mix_clip_lengths() -> typing.Any:
    adapter = small_adapter().eval()
    open_gates(adapter)
    c, _, _, _ = inputs(B=2)
    lengths = [8, 16]                                    # e.g. K = 1 vs K = 2 tubelets
    f = [torch.randn(n, D) for n in lengths]
    m = [torch.rand(n) for n in lengths]
    dt = [torch.randint(-2, 3, (n,)) for n in lengths]
    batch = pad_video_batch(f, m, dt)
    assert batch["kv_padding_mask"].sum(1).tolist() == [8, 0]
    out = adapter(c, batch["tokens"], batch["coverage"], batch["dt"], batch["kv_padding_mask"])
    for i in range(2):
        alone = adapter(c[i:i + 1], f[i][None], m[i][None], dt[i][None])
        torch.testing.assert_close(out[i:i + 1], alone, atol=1e-5, rtol=1e-5)


def test_attention_mass_on_object() -> typing.Any:
    adapter = small_adapter()
    c, f, _, dt = inputs()
    ones, zeros = torch.ones(2, 12), torch.zeros(2, 12)
    assert attention_mass_on_object(adapter, c, f, ones, dt) == pytest.approx([1.0, 1.0])
    assert attention_mass_on_object(adapter, c, f, zeros, dt) == pytest.approx([0.0, 0.0])
    half = torch.cat([ones[:, :6], zeros[:, 6:]], 1)
    assert all(0 < x < 1 for x in attention_mass_on_object(adapter, c, f, half, dt))


def test_time_offsets_out_of_range_fail() -> typing.Any:
    adapter = small_adapter()
    c, f, m, dt = inputs()
    with pytest.raises(AssertionError):
        adapter(c, f, m, dt + 10)
    small_adapter(use_time_embedding=False)(c, f, m, dt + 10)   # no table, no constraint
