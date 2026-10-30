"""NAVI preparation using the same SAM wrapper as synthetic training."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import chz
import numpy as np
from PIL import Image, ImageOps
import torch
from .config import ExperimentConfig
from .data.navi import Example, read_manifest
from .models.sam3d import build_sam3d
from .storage import atomic_json, atomic_torch

def visibility(points: np.ndarray, annotation: dict[str, Any], scene: Path) -> np.ndarray:
    # Reuse NAVI's published camera/depth convention instead of guessing q/t axes.
    import data_util as navi
    transform, projection = navi.camera_matrices_from_annotation(annotation)
    # NAVI mixes NumPy float64 transforms with Torch float32 intrinsics.
    transform = np.asarray(transform, dtype=np.float64)
    projection = np.asarray(projection, dtype=np.float64)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    camera = (transform @ homogeneous.T).T
    clip = (projection @ camera.T).T
    ndc = clip[:, :3] / np.maximum(clip[:, 3:4], 1e-12)
    height, width = annotation["image_size"]
    px = np.round((ndc[:, 0] + 1) * width / 2).astype(int)
    # NAVI tutorial maps both NDC axes with (coordinate + 1) / 2.
    py = np.round((1 + ndc[:, 1]) * height / 2).astype(int)
    depth_path = scene / "depth" / (Path(annotation["filename"]).stem + ".png")
    depth = navi.read_depth_from_png(str(depth_path))
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height) & (clip[:, 3] > 0)
    sampled = depth[np.clip(py, 0, height - 1), np.clip(px, 0, width - 1)]
    # NAVI projection has clip.w = camera.z, so forward depth is positive Z.
    expected = camera[:, 2]
    return inside & (np.abs(sampled - expected) < np.maximum(np.abs(expected) * 0.01, 1e-3))

def prepare(config: ExperimentConfig) -> None:
    import imageio.v2 as imageio
    import trimesh
    if not torch.cuda.is_available():
        raise RuntimeError("Dataset encoding belongs on the remote allocation")
    root, output = Path(config.paths.data_root), Path(config.data.prepared)
    output.mkdir(parents=True, exist_ok=True)
    prepared_data = chz.asdict(config.data)
    del prepared_data["video_input"]  # Runtime pixel preprocessing does not change cached native features or geometry.
    signature = {"schema_version": 3, "data": prepared_data, "sam_weights": config.sam3d.hf_repo, "sam_revision": config.sam3d.checkpoint_revision, "sam_commit": config.provenance.sam_commit, "coordinates": "z-up-aabb-unit-cube-surface64-v1", "visibility": "navi-positive-z-positive-image-y-v2"}
    metadata = output / "preparation.json"
    if (output / "manifest.jsonl").is_file():
        if not metadata.is_file() or json.loads(metadata.read_text())["signature"] != signature:
            raise ValueError("Prepared dataset configuration changed; choose a new prepared directory, preserving existing data")
        read_manifest(output / "manifest.jsonl")
        return
    quality: list[dict[str, Any]] = []
    objects = sorted({path.parent.parent.name for path in root.rglob("annotations.json") if path.parent.name.startswith("video")})
    required = config.data.train_objects + config.data.val_objects + config.data.test_objects
    if len(objects) < required:
        raise ValueError(f"Expected {required} objects with video; found {len(objects)}")
    order = np.random.default_rng(config.seed).permutation(objects).tolist()
    splits = {name: "train" if index < config.data.train_objects else "val" if index < config.data.train_objects + config.data.val_objects else "test" for index, name in enumerate(order)}
    atomic_json(output / "object_splits.json", splits)
    model = build_sam3d(config, torch.device("cuda"))
    rows: list[Example] = []
    for object_id in order:
        split = splits[object_id]
        # The two-hour pilot prepares two training and two validation objects only.
        limit = config.data.pilot_train_objects if split == "train" else config.data.pilot_val_objects
        allowed = [name for name in order if splits[name] == split][:limit]
        if split == "test" or object_id not in allowed:
            continue
        scenes = sorted(path.parent for path in root.rglob("annotations.json") if path.parent.parent.name == object_id and path.parent.name.startswith("video"))[:config.data.clips_per_object]
        object_root = scenes[0].parent
        source = object_root / "3d_scan" / f"{object_id}.glb"
        if not source.exists():
            source = object_root / "3d_scan" / f"{object_id}.obj"
        mesh = trimesh.load(source, force="mesh", process=False)
        original_points, _ = trimesh.sample.sample_surface(mesh, 4096, seed=42)
        vertices = np.array(mesh.vertices)
        rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        vertices = vertices @ rotation.T
        center = (vertices.max(0) + vertices.min(0)) / 2
        extent = float(np.ptp(vertices, axis=0).max())
        if extent <= 0:
            raise ValueError("Degenerate mesh")
        mesh.vertices = np.clip((vertices - center) / extent, -0.5 + 1e-6, 0.5 - 1e-6)
        canonical_points = (original_points @ rotation.T - center) / extent
        object_dir = output / object_id
        object_dir.mkdir(exist_ok=True)
        gt = object_dir / "ground_truth.glb"
        mesh.export(gt)
        voxels = mesh.voxelized(pitch=1 / 64).points
        indices = np.clip(np.floor((voxels + 0.5) * 64).astype(int), 0, 63)
        occupancy = torch.zeros(1, 1, 64, 64, 64, device="cuda")
        occupancy[0, 0, indices[:, 0], indices[:, 1], indices[:, 2]] = 1
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = {"shape": model.encode_voxels(occupancy[:, 0])}
            reconstruction = model.decode_latent(encoded["shape"])[:, None]
        if not torch.isfinite(encoded["shape"]).all() or not torch.isfinite(reconstruction).all():
            raise FloatingPointError("Non-finite upstream VAE round trip")
        reconstructed = reconstruction > 0
        if not reconstructed.any() or reconstructed.all():
            raise ValueError("Degenerate upstream VAE reconstruction; inspect occupancy convention before training")
        overlap = (reconstructed & (occupancy > 0)).sum().item()
        union = (reconstructed | (occupancy > 0)).sum().item()
        quality.append({"object_id": object_id, "vae_surface_occupancy_iou": overlap / max(union, 1)})
        occupancy_path = object_dir / "ground_truth_occupancy.npy"
        np.save(occupancy_path, occupancy[0, 0].cpu().numpy().astype(bool))
        target = object_dir / "target.pt"
        atomic_torch(target, {key: value[0].float().cpu() for key, value in encoded.items()})
        for scene in scenes:
            annotations = json.loads((scene / "annotations.json").read_text())
            selected = [annotations[i] for i in np.linspace(0, len(annotations) - 1, config.data.frames).round().astype(int)]
            frames = [str(scene / "images" / item["filename"]) for item in selected]
            masks = [str(scene / "masks" / (Path(item["filename"]).stem + ".png")) for item in selected]
            directory = object_dir / scene.name
            directory.mkdir(exist_ok=True)
            video_path = directory / "input.mp4"
            with imageio.get_writer(video_path, fps=4, codec="libx264", macro_block_size=1) as writer:
                for path in frames:
                    writer.append_data(np.array(ImageOps.exif_transpose(Image.open(path)).convert("RGB").resize((512, 512))))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                rgb = torch.from_numpy(np.array(ImageOps.exif_transpose(Image.open(frames[0])).convert("RGB")))[None].cuda()
                mask = torch.from_numpy(np.asarray(Image.open(masks[0]).convert("L")) > 0)[None].cuda()
                tokens = torch.cat(list(model.condition(rgb, mask).values()), dim=1)
            token_path = directory / "reference_tokens.pt"
            torch.save(tokens[0].float().cpu(), token_path)
            observed = [visibility(original_points, item, scene) for item in selected]
            any_observed = np.logical_or.reduce(observed)
            if not any_observed.any():
                raise ValueError(f"No scan points visible in {object_id}/{scene.name}; check camera/depth alignment")
            quality.append({"object_id": object_id, "scene": scene.name, "reference_visible_points": int(observed[0].sum()), "newly_revealed_points": int((any_observed & ~observed[0]).sum()), "never_observed_points": int((~any_observed).sum())})
            np.savez(directory / "visibility.npz", points=canonical_points, reference_visible=observed[0], newly_revealed=any_observed & ~observed[0], never_observed=~any_observed)
            example_id = object_id + "-" + hashlib.sha256(scene.name.encode()).hexdigest()[:10]
            rows.append(Example(example_id=example_id, object_id=object_id, split=split, frames=frames, masks=masks, video=str(video_path), target=str(target), ground_truth_mesh=str(gt), reference_tokens=str(token_path), visibility=str(directory / "visibility.npz"), occupancy=str(occupancy_path)))
    atomic_json(metadata, {"signature": signature, "quality": quality})
    temporary = output / "manifest.jsonl.tmp"
    temporary.write_text("\n".join(row.model_dump_json() for row in rows) + "\n")
    temporary.replace(output / "manifest.jsonl")
