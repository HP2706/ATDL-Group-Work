"""T3a: V-JEPA 2.1 token order and image tokenizer; SAM 3D's DINOv2 for C1/C2."""

import pytest
import torch

from vidsam3d.models.encoders import DinoEncoder

pytestmark = pytest.mark.gpu


def test_vjepa_token_counts(vjepa):
    for K, n in ((1, 256), (4, 512), (16, 2048)):
        out = vjepa.encode(torch.rand(1, K, 3, 256, 256, device="cuda"))
        assert out.shape == (1, n, 1024), (K, out.shape)


@pytest.mark.parametrize("t, h, w", [(0, 0, 0), (1, 3, 5), (3, 15, 2)])
def test_t3a_vjepa_token_order_is_t_h_w(vjepa, t, h, w):
    g = torch.Generator(device="cuda").manual_seed(0)
    clip = torch.rand(1, 8, 3, 256, 256, generator=g, device="cuda")
    changed = clip.clone()
    changed[0, 2 * t:2 * t + 2, :, 16 * h:16 * h + 16, 16 * w:16 * w + 16] = 1 - changed[
        0, 2 * t:2 * t + 2, :, 16 * h:16 * h + 16, 16 * w:16 * w + 16]
    diff = (vjepa.encode(changed) - vjepa.encode(clip)).norm(dim=-1)[0]
    assert int(diff.argmax()) == t * 256 + h * 16 + w


def test_sam3d_dinov2_grid(sam3d):
    dino = DinoEncoder(sam3d.dino_backbone(), 224)
    out = dino.encode(torch.rand(1, 2, 3, 224, 224, device="cuda"))
    assert out.shape[:2] == (1, 2 * 256) and dino.patch == 14
