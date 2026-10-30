"""Portable setup and one-clip commands; generation itself lives in video3d."""
from __future__ import annotations

from functools import partial
import json
from pathlib import Path

import chz
import fire
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import requests

from video3d.data.generation.config import Config
from video3d.data.generation.generate import generate_clip


def stage_source(source: Path, directory: Path) -> Path:
    return source


def prepare(root: str) -> None:
    from video3d.data.generation.assets import Asset, preprocess
    from video3d.data.generation.assetfilter import accepted_assets
    destination = Path(root)
    source_root = Path("/sources/runs/paired-pilot-2026-10-22-00-52-54/assets")
    meshes = [path for path in sorted(source_root.glob("*/object.glb")) if 10000 < path.stat().st_size < 20000000][:12]
    if len(meshes) < 4:
        raise ValueError("Expected at least four cached source object meshes")
    cfg = chz.replace(Config(), dataset_root=root, bg_pool_dir=str(destination / "backgrounds"),
        blender_custom_path="/opt/blender/blender-4.5.15-linux-x64", blender_install_path="/tmp/blender",
        n_objects_choices=[2], n_objects_probs=[1.0], cycles_samples=16, denoiser="INTEL", render_devices=["CUDA"],
        arc_span_range_deg=(55., 70.), elevation_drift_range_deg=(0., 0.),
        orbit_factor_range=(1.1, 1.3), occlusion_mode="recoverable", max_retries=12,
        filter_extent_min=0.25, filter_normal_eig_min=0.04)
    assets = [Asset(asset_id=path.parent.name, stage=partial(stage_source, path),
                    meta={"source": "objaverse", "source_path": str(path), "source_name": path.parent.name}) for path in meshes]
    preprocess(cfg, assets, "train", workers=1)
    accepted = accepted_assets(cfg, "train")
    if len(accepted) < 2:
        raise ValueError(f"Too few compact assets after existing geometric filter: {accepted}")
    backgrounds = destination / "backgrounds"
    backgrounds.mkdir(exist_ok=True)
    background = backgrounds / "photograph.jpg"
    url = "https://images.unsplash.com/photo-1600210492486-724fe5c67fb0?w=2048&q=90"
    if not background.exists():
        response = requests.get(url, headers={"User-Agent": "ATDL-research-occlusion-pilot/1.0"}, timeout=60)
        response.raise_for_status()
        raw = backgrounds / "source.jpg"
        raw.write_bytes(response.content)
        with Image.open(raw) as image:
            image.convert("RGB").resize((2048, 1024)).save(background, quality=95)
    (backgrounds / "train.jsonl").write_text(json.dumps({"image_id": "unsplash-1600210492486", "path": str(background), "source_url": url}) + "\n")
    (destination / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (destination / "preparation.json").write_text(json.dumps({"cached_source": str(source_root),
        "source_meshes": [str(p) for p in meshes], "accepted_assets": accepted, "background_url": url,
        "scope": "small generation pilot with cached Objaverse objects and one photographic background; not a representative training dataset"}, indent=2))
    print(json.dumps({"accepted_assets": len(accepted), "config": cfg.to_dict()}), flush=True)


def generate(root: str, seed: int) -> None:
    destination = Path(root)
    cfg = Config.load(destination / "config.json")
    clip_path = destination / "clips/train" / f"{seed:08d}"
    if not (clip_path / "meta.json").exists():
        generate_clip(cfg, "train", seed, destination)
    meta = json.loads((clip_path / "meta.json").read_text())
    preview = destination / "previews"
    preview.mkdir(exist_ok=True)
    with imageio.get_writer(preview / f"{seed:08d}.mp4", fps=8, codec="libx264", macro_block_size=1) as writer:
        for frame in sorted((clip_path / "rgb").glob("*.png")):
            writer.append_data(np.asarray(Image.open(frame).convert("RGB")))
    reference = cfg.num_frames // 2
    montage = Image.new("RGB", (512 * 3, 544), "white")
    draw = ImageDraw.Draw(montage)
    for column, t in enumerate((0, reference, cfg.num_frames - 1)):
        montage.paste(Image.open(clip_path / "rgb" / f"{t:03d}.png").convert("RGB"), (column * 512, 32))
        occlusion = 1 - meta["visibility"]["visible_frac"][t][0]
        draw.text((column * 512 + 8, 8), f"frame {t}, target occlusion {occlusion:.1%}", fill="black")
    montage.save(preview / f"{seed:08d}.jpg", quality=90)
    (preview / f"{seed:08d}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"seed": seed, "occlusion": meta["occlusion"]}), flush=True)


def finalize(root: str) -> None:
    from video3d.data.generation.export import export, verify
    from video3d.data.generation.occlusion import occlusion_failures
    destination = Path(root)
    cfg = Config.load(destination / "config.json")
    paths = sorted((destination / "clips/train").glob("*/meta.json"))
    if len(paths) != 10:
        raise ValueError(f"Expected 10 completed clips, found {len(paths)}")
    entries = []
    for path in paths:
        meta = json.loads(path.read_text())
        failures = occlusion_failures(np.asarray(meta["visibility"]["visible_frac"]), cfg)
        if failures or meta["occlusion"]["external_occlusion_recovery"] < cfg.occlusion_min_recovery:
            raise ValueError(f"Invalid accepted clip {meta['clip_id']}: {failures}")
        entries.append({"clip_id": meta["clip_id"], "split": "train", "N": meta["num_objects"],
            "arc_span_deg": meta["arc_span_deg"], "seed": meta["seed"], "path": f"clips/train/{meta['clip_id']}"})
    (destination / "manifest.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    out = destination / "export/train"
    index = export(destination, "train", out, clips_per_shard=5, workers=2)
    verify(destination, "train", out, 10)
    report = {"completed_clips": len(paths), "frames": len(paths) * cfg.num_frames,
              "acceptance_rechecked": True, "export_roundtrip_verified": True,
              "export_index": str(out / "index.json"), "export_shards": len(index["shards"])}
    (destination / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    fire.Fire({"prepare": prepare, "generate": generate, "finalize": finalize})
