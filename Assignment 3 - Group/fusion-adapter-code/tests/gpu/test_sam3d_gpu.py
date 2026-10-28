"""T1 (identity at init) and the SAM 3D facts the wrapper relies on, on the real model."""

import pytest
import torch

from vidsam3d.inspect_sam3d import numerical_checks
from vidsam3d.models.adapter import VideoFusionAdapter
from vidsam3d.models.sam3d import apply_adapter

pytestmark = pytest.mark.gpu


def test_condition_sets_split_and_are_deterministic(sam3d, eval_sample, cond):
    x = eval_sample["inputs"]
    again = sam3d.condition(x["ref_rgb"][None].cuda(), x["ref_mask"][None].cuda())
    assert list(cond) == sam3d.set_names and len(cond) >= 2
    assert all(torch.equal(cond[k], again[k]) for k in cond)


def test_t1_adapted_pipeline_at_init_equals_sam3d(gpu_cfg, sam3d, cond):
    width = next(iter(cond.values())).shape[-1]
    adapter = VideoFusionAdapter(d_cond=width, d_vid=1024).cuda()
    f, m = torch.randn(1, 64, 1024, device="cuda"), torch.rand(1, 64, device="cuda")
    adapted = apply_adapter(cond, adapter, f, m, torch.zeros(1, 64, dtype=torch.long, device="cuda"),
                            list(cond))
    assert all(torch.equal(adapted[k], cond[k]) for k in cond)
    plain = [sam3d.decode_latent(sam3d.sample(cond, [7], gpu_cfg.sampler)[0]) for _ in range(2)]
    if not torch.equal(*plain):
        pytest.skip("sampling is not bitwise deterministic on this GPU; T1 cannot be checked exactly")
    ours = sam3d.decode_latent(sam3d.sample(adapted, [7], gpu_cfg.sampler)[0])
    assert torch.equal(ours, plain[0])


def test_null_branch_and_layout_independence(sam3d, cond):
    checks = numerical_checks(sam3d, cond)
    assert sam3d.backbone.force_zeros_cond, "CFG null branch is not zeros"
    assert checks["3 max |null branch(cond) - null branch(random cond)|"] == 0
    assert checks["7 max |v_shape(layout a) - v_shape(layout b)|"] == 0


def test_gradients_reach_conditioning_tokens(sam3d, cond):
    c = {k: v.detach().clone().requires_grad_(True) for k, v in cond.items()}
    x = torch.randn(1, *sam3d.latent_shape, device="cuda")
    layout = {k: torch.randn(1, *s, device="cuda") for k, s in sam3d.layout_shape.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        v, _ = sam3d.velocity(x, torch.full((1,), 0.3, device="cuda"), c, layout)
    v.float().pow(2).mean().backward()
    assert all(t.grad is not None and t.grad.abs().sum() > 0 for t in c.values())
    assert all(p.grad is None for p in sam3d.parameters())
