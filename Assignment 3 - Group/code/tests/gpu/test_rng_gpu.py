from __future__ import annotations
from pathlib import Path
import pytest
import torch
from video3d.storage import rng_state, restore_rng

pytestmark = pytest.mark.gpu


def test_rng_restore_after_cuda_checkpoint_load(tmp_path: Path) -> None:
    path=tmp_path/"rng.pt"
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.save(rng_state(),path)
    cpu_expected=torch.rand(3)
    cuda_expected=torch.rand(3,device="cuda")
    loaded=torch.load(path,map_location="cuda",weights_only=False)
    assert loaded["torch"].is_cuda
    restore_rng(loaded)
    assert torch.equal(torch.rand(3),cpu_expected)
    assert torch.equal(torch.rand(3,device="cuda"),cuda_expected)


def test_fusion_accepts_native_half_conditioning_without_autocast() -> None:
    from video3d.models.adapter import VideoFusionAdapter
    from video3d.models.sam3d import apply_adapter
    adapter=VideoFusionAdapter(d_cond=32,d_vid=32,heads=4).cuda()
    condition={"image":torch.randn(1,4,32,device="cuda",dtype=torch.float16),
               "mask":torch.randn(1,4,32,device="cuda",dtype=torch.float16)}
    output=apply_adapter(condition,adapter,torch.randn(1,8,32,device="cuda"),
                         torch.ones(1,8,device="cuda"),torch.zeros(1,8,device="cuda",dtype=torch.long),["image"])
    assert torch.isfinite(output["image"]).all()
    assert torch.equal(output["image"],condition["image"])
    assert output["mask"] is condition["mask"]
