"""Exercise the existing generator with one small, locally constructed asset on a remote GPU."""
from __future__ import annotations
import json
from pathlib import Path
import chz
import fire
import numpy as np
from PIL import Image
import trimesh
from video3d.data.generation.assets import Asset, preprocess
from video3d.data.generation.config import Config
from video3d.data.generation.generate import generate_clip, update_manifest
from video3d.data.generation.export import export, verify


def check(output: str) -> None:
    root=Path(output)
    root.mkdir(parents=True,exist_ok=True)
    raw=root/"box.glb"
    trimesh.creation.box(extents=[1.0,0.6,0.4]).export(raw)
    backgrounds=root/"backgrounds"
    backgrounds.mkdir(exist_ok=True)
    background=backgrounds/"fixture.jpg"
    x,y=np.meshgrid(np.linspace(0,1,1024),np.linspace(0,1,512))
    pixels=np.stack([80+100*x,90+90*y,140+30*np.sin(2*np.pi*x)*np.cos(np.pi*y)],axis=-1)
    Image.fromarray(pixels.astype(np.uint8)).save(background)
    (backgrounds/"eval.jsonl").write_text(json.dumps({"image_id":"fixture","path":str(background.resolve())})+"\n")
    cfg=chz.replace(Config(),dataset_root=str(root),bg_pool_dir=str(backgrounds),
                    blender_install_path="/cache/blender",width=128,height=128,num_frames=4,
                    blender_custom_path="/opt/blender/blender-4.5.15-linux-x64",
                    cycles_samples=2,denoiser="INTEL",n_objects_choices=[1],n_objects_probs=[1.0])
    (root/"config.json").write_text(json.dumps(cfg.to_dict(),indent=2))
    asset=Asset(asset_id="box",stage=lambda directory:raw.resolve(),meta={"source":"fixture"})
    preprocess(cfg,[asset],"eval",workers=1)
    if not (root/"assets/eval/box/meta.json").exists():
        raise RuntimeError("Generator asset preparation failed")
    if not (root/"clips/eval/00000000/meta.json").exists():
        update_manifest(root,generate_clip(cfg,"eval",0,root))
    out=root/"export/eval"
    index=export(root,"eval",out,clips_per_shard=1)
    verify(root,"eval",out,1)
    if index["num_clips"] != 1:
        raise AssertionError("Generator export did not contain the rendered clip")
    print(json.dumps({"rendered_clips":1,"export_verified":True}))


if __name__ == "__main__":
    fire.Fire(check)
