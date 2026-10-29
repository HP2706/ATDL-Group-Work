from __future__ import annotations
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch
from tests.fixtures.fake_config import FAKE_OVERRIDES
from video3d.config import load_config
from video3d.data import build_dataset
from video3d.data.navi import Example
from video3d.models.fake_sam3d import FakeSam3D
from video3d.train import train
from video3d.evaluate import evaluate


def test_navi_uses_shared_trainer_and_evaluator(tmp_path: Path) -> None:
    prepared=tmp_path/"prepared"
    prepared.mkdir()
    image=prepared/"frame.png"
    mask=prepared/"mask.png"
    Image.fromarray(np.full((64,64,3),120,dtype=np.uint8)).save(image)
    foreground=np.zeros((64,64),dtype=np.uint8)
    foreground[16:48,16:48]=255
    Image.fromarray(foreground).save(mask)
    vox=torch.zeros(1,64,64,64,dtype=torch.bool)
    vox[:,16:48,16:48,16:48]=True
    np.save(prepared/"occupancy.npy",vox[0].numpy())
    latent=FakeSam3D().encode_voxels(vox)[0]
    torch.save({"shape":latent},prepared/"target.pt")
    rows=[]
    for split in ["train","val","eval"]:
        rows.append(Example(example_id=split,object_id=split,split=split,
            frames=[str(image)]*4,masks=[str(mask)]*4,video="unused.mp4",
            target=str(prepared/"target.pt"),ground_truth_mesh="unused.glb",
            reference_tokens="unused.pt",occupancy=str(prepared/"occupancy.npy")))
    (prepared/"manifest.jsonl").write_text("\n".join(row.model_dump_json() for row in rows)+"\n")
    overrides={**FAKE_OVERRIDES,"data.source":"navi","data.prepared":str(prepared),
               "data.frames":4,"train.max_steps":2,"run_name":"navi-smoke",
               "paths.output_root":str(tmp_path/"out"),"paths.scratch_root":str(tmp_path/"scratch")}
    cfg=load_config("c3",overrides=[f"{key}={value}" for key,value in overrides.items()])
    sample=build_dataset(cfg,"train",True)[0]
    assert sample["inputs"]["frames"].shape == (4,3,32,32)
    assert sample["target"].shape == latent.shape
    assert train(cfg)["step"] == 2
    result=evaluate(cfg,str(tmp_path/"out/navi-smoke"),k=4,limit=1)
    assert json.loads(result.read_text().splitlines()[0])["clip_id"] == "eval"


@pytest.mark.parametrize("numpy_transform",[False,True])
def test_visibility_preserves_published_navi_convention(numpy_transform: bool,
                                                       monkeypatch: pytest.MonkeyPatch,
                                                       tmp_path: Path) -> None:
    from video3d.prepare import visibility
    transform=np.eye(4) if numpy_transform else torch.eye(4,dtype=torch.float64)
    projection=torch.eye(4,dtype=torch.float32)
    projection[3,2],projection[3,3]=1.0,0.0
    depth=np.zeros((4,4),dtype=np.float32)
    depth[2,2],depth[3,2]=1.0,1.0
    reference=SimpleNamespace(camera_matrices_from_annotation=lambda annotation:(transform,projection),
                              read_depth_from_png=lambda path:depth)
    monkeypatch.setitem(sys.modules,"data_util",reference)
    points=np.array([[0,0,1],[0,.5,1],[0,0,-1],[3,0,1]],dtype=np.float64)
    assert visibility(points,{"image_size":[4,4],"filename":"frame.jpg"},tmp_path).tolist() == [True,True,False,False]


def test_full_scene_shared_rgb_mask_letterboxing() -> None:
    from video3d.config import DataConfig, CropConfig
    from video3d.data.samples import preprocess_video
    image=np.zeros((16,32,3),dtype=np.uint8)
    image[:,:,0]=255
    mask=np.zeros((16,32),dtype=bool)
    mask[:,:16]=True
    cfg=DataConfig(video_input='full_scene',crop=CropConfig(size=32))
    frames,masks,_=preprocess_video([image],[mask],cfg)
    assert masks[:,8:24,:16].all()
    assert masks[:,:8].sum()==0 and masks[:,:,16:].sum()==0
    torch.testing.assert_close(frames[0,:,10,25],torch.tensor([1.,0.,0.]))
