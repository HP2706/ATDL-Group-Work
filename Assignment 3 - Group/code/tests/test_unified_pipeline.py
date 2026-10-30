"""Both strategies and LoRA combinations use the same trainer and checkpoint format."""
from __future__ import annotations
import typing
import json
from pathlib import Path
import pytest
import torch
from tests.fixtures.fake_config import fake_train_cfg
from video3d.config import replace_config
from video3d.train import train
from video3d.evaluate import evaluate


@pytest.mark.parametrize("strategy", ["fusion", "pooler"])
@pytest.mark.parametrize("video_lora,sam_lora", [(False,False),(True,False),(False,True),(True,True)])
def test_shared_trainer_and_resume(fake_root: typing.Any, tmp_path: typing.Any, strategy: str,
                                  video_lora: bool, sam_lora: bool) -> None:
    overrides = {"adapter.strategy":strategy, "train.max_steps":4,
                 "adapter.pooler_queries":4, "adapter.foreground_threshold":0.01,
                 "encoder.vjepa.video_lora":video_lora,"sam3d.lora":sam_lora,
                 "train.ckpt_every":2}
    cfg = fake_train_cfg(fake_root,tmp_path,**overrides)
    train(cfg,stop_at=2)
    assert train(cfg)["step"] == 4
    directory = Path(cfg.paths.output_root)/cfg.run_name
    state = torch.load(directory/"ckpt_4.pt",weights_only=False)
    assert state["schema_version"] == 3
    assert bool(state["extra"]) == (video_lora or sam_lora)
    assert all(torch.isfinite(tensor).all() for tensor in state["extra"].values())
    changed = replace_config(cfg,**{"optim.lr":0.02})
    with pytest.raises(ValueError,match="mismatch"):
        train(changed)
    path = evaluate(cfg,str(directory),limit=1)
    record=json.loads(path.read_text().splitlines()[0])
    assert record["schema_version"] == 3 and record["sample_id"]
    media = path.parent / "media" / cfg.run_name / record["sample_id"].replace("/","-")
    assert (media/"input.mp4").is_file() and (media/"ground_truth.glb").is_file()
