"""Frozen DINOv2 frame embeddings use the shared attentive conditioning path."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from video3d.config import AdapterConfig
from video3d.models.encoders import DinoEncoder, kv_tokens
from video3d.models.pooler import MaskPooler


class PatchEmbed(nn.Module):
    patch_size: tuple[int, int] = (14, 14)


class DinoFeatureFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.projection = nn.Linear(3 * 14 * 14, 32)

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        patches = F.unfold(images, 14, stride=14).transpose(1, 2)
        tokens = self.projection(patches)
        return {'x_norm_patchtokens':tokens,
                'x_norm_clstoken':tokens.new_full((len(images), 32), 10000),
                'x_norm_regtokens':tokens.new_full((len(images), 4, 32), 20000)}


def test_dinov2_all_frames_align_with_foreground_without_special_tokens() -> None:
    encoder = DinoEncoder(DinoFeatureFixture(), image_size=28)
    frames = torch.stack([torch.full((3, 28, 28), i / 16) for i in range(16)])
    masks = torch.zeros(16, 28, 28)
    masks[::2, :14, :14] = 1
    masks[1::2, 14:, 14:] = 1
    inputs = {'frames':frames, 'frame_masks':masks, 'dt':torch.zeros(8), 'ref_pos':7}
    tokens, coverage, dt = kv_tokens(encoder, inputs, 'dinov2_frames')
    assert tokens.shape == (16 * 4, 32)
    assert coverage.reshape(16, 4).argmax(1).tolist() == [0, 3] * 8
    assert dt.reshape(16, 4)[:, 0].tolist() == list(range(-7, 9))
    for frame in [0, 7, 15]:
        assert torch.allclose(tokens.reshape(16, 4, 32)[frame],
                              encoder.encode(frames[frame:frame + 1][None])[0])
    assert not tokens.requires_grad
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert tokens.abs().max() < 10000


def test_dinov2_pooler_trains_while_native_backbone_stays_frozen() -> None:
    encoder = DinoEncoder(DinoFeatureFixture(), image_size=28)
    frames = torch.rand(1, 4, 3, 28, 28)
    tokens = encoder.encode(frames)
    model = MaskPooler(AdapterConfig(d_vid=32, d_cond=16, heads=4, pooler_queries=4))
    cond = {'rgb':torch.randn(1, 8, 16), 'mask':torch.randn(1, 8, 16),
            'pointmap':torch.randn(1, 8, 16)}
    result = model.condition(cond, tokens, torch.ones(1, 16), ['rgb'])
    result['rgb'].square().mean().backward()
    assert result['rgb'].shape == (1, 4, 16)
    assert result['mask'] is cond['mask'] and result['pointmap'] is cond['pointmap']
    assert model.pooler.query_tokens.grad is not None
    assert model.pooler.query_tokens.grad.abs().sum() > 0
    assert model.projector.weight.grad is not None
    assert model.projector.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in encoder.parameters())
