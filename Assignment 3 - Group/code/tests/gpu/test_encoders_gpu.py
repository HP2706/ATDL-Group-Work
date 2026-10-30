"""T3a: V-JEPA 2.1 token order and image tokenizer; SAM 3D's DINOv2 for C1/C2."""
import typing

import pytest
import torch

from video3d.models.encoders import DinoEncoder

pytestmark = pytest.mark.gpu


def test_vjepa_token_counts(vjepa: typing.Any) -> typing.Any:
    for K, n in ((1, 256), (4, 512), (16, 2048)):
        out = vjepa.encode(torch.rand(1, K, 3, 256, 256, device="cuda"))
        assert out.shape == (1, n, 1024), (K, out.shape)


@pytest.mark.parametrize("t, h, w", [(0, 0, 0), (1, 3, 5), (3, 15, 2)])
def test_t3a_vjepa_token_order_is_t_h_w(vjepa: typing.Any, t: typing.Any, h: typing.Any, w: typing.Any) -> typing.Any:
    g = torch.Generator(device="cuda").manual_seed(0)
    clip = torch.rand(1, 8, 3, 256, 256, generator=g, device="cuda")
    changed = clip.clone()
    changed[0, 2 * t:2 * t + 2, :, 16 * h:16 * h + 16, 16 * w:16 * w + 16] = 1 - changed[
        0, 2 * t:2 * t + 2, :, 16 * h:16 * h + 16, 16 * w:16 * w + 16]
    # Self-attention spreads changes; patch embeddings have exact local support.
    captured: list[torch.Tensor] = []
    def capture(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured.append(output.detach().clone())
    handle = vjepa.model.patch_embed.register_forward_hook(capture)
    try:
        vjepa.encode(clip)
        vjepa.encode(changed)
    finally:
        handle.remove()
    assert len(captured) == 2
    diff = (captured[1].float() - captured[0].float()).norm(dim=-1)[0]
    assert int(diff.argmax()) == t * 256 + h * 16 + w
    assert int((diff > 1e-5).sum()) == 1


def test_sam3d_dinov2_grid(sam3d: typing.Any) -> typing.Any:
    dino = DinoEncoder(sam3d.dino_backbone(), 224)
    out = dino.encode(torch.rand(1, 2, 3, 224, 224, device="cuda"))
    assert out.shape[:2] == (1, 2 * 256) and dino.patch == 14
