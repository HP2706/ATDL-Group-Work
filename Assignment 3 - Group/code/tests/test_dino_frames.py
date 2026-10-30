"""The learned DINO pooler receives independent per-frame spatial tokens."""
import torch
from video3d.config import load_config
from video3d.models.encoders import FakeEncoder, kv_tokens


def test_per_frame_dino_tokens_and_masks_are_not_pair_averaged() -> None:
    encoder = FakeEncoder(d=8, image_size=32, patch=16, tubelet=1)
    frames = torch.stack([torch.full((3, 32, 32), value) for value in (0.0, 0.2, 0.8, 1.0)])
    masks = torch.ones(4, 32, 32)
    masks[1] = 0
    inputs = {"frames": frames, "frame_masks": masks, "dt": torch.tensor([-1, 0]), "ref_pos": 2}
    tokens, coverage, offsets = kv_tokens(encoder, inputs, "dinov2_frames")
    assert tokens.shape == (16, 8)
    assert torch.equal(tokens, encoder.encode(frames[None])[0])
    assert torch.equal(coverage, torch.tensor([1., 0., 1., 1.]).repeat_interleave(4))
    assert torch.equal(offsets, torch.tensor([-2, -1, 0, 1]).repeat_interleave(4))
    paired, pair_coverage, _ = kv_tokens(encoder, inputs, "dinov2_clip")
    assert paired.shape == (8, 8)
    assert torch.equal(paired, tokens.view(2, 2, 4, 8).mean(1).flatten(0, 1))
    assert torch.equal(pair_coverage, torch.tensor([0.5, 1.0]).repeat_interleave(4))


def test_dino_pooler_preset_keeps_shared_training_inputs() -> None:
    cfg = load_config("dino_pooler", "local")
    assert cfg.adapter.strategy == "pooler"
    assert cfg.encoder.kv_source == "dinov2_frames"
    assert not cfg.encoder.dinov2.pair_average
    assert not cfg.sam3d.lora and not cfg.encoder.vjepa.video_lora
    assert cfg.data.video_input == "object_crop"
