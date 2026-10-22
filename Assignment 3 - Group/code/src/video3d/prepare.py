from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any
import chz
import numpy as np
from PIL import Image, ImageOps
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from .config import ExperimentConfig
from .data import Example, read_manifest
from .model import NativeSAM, load_component
from .storage import atomic_json, atomic_torch


def image_tensor(image: Image.Image, size: int, nearest: bool = False) -> torch.Tensor:
    image = image.resize((size, size), Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR)
    array = np.array(image, copy=True)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    return torch.from_numpy(array).permute(2, 0, 1).float()[None].cuda() / 255


def reference_inputs(image_path: Path, mask_path: Path) -> dict[str, torch.Tensor]:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    y, x = np.nonzero(np.asarray(mask) > 0)
    if not len(x):
        raise ValueError(f"Empty mask: {mask_path}")
    cx, cy = (int(x.min()) + int(x.max())) / 2, (int(y.min()) + int(y.max())) / 2
    side = max(int(x.max() - x.min() + 1), int(y.max() - y.min() + 1)) * 1.2
    crop_box = tuple(round(value) for value in [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2])
    image_crop, mask_crop = image.crop(crop_box), mask.crop(crop_box)
    full_image = ImageOps.pad(image, (max(image.size), max(image.size)))
    full_mask = ImageOps.pad(mask, full_image.size)
    return {"image": image_tensor(image_crop, 518), "mask": image_tensor(mask_crop, 518, True), "rgb_image": image_tensor(full_image, 518), "rgb_image_mask": image_tensor(full_mask, 518, True), "pointmap": torch.zeros(1, 3, 256, 256, device="cuda"), "rgb_pointmap": torch.zeros(1, 3, 256, 256, device="cuda")}


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
    sys.path.insert(0, "/work/dependencies/navi")
    root, output = Path(config.data.root), Path(config.data.prepared)
    output.mkdir(parents=True, exist_ok=True)
    prepared_data = chz.asdict(config.data)
    del prepared_data["video_input"]  # Runtime pixel preprocessing does not change cached native features or geometry.
    signature = {"schema_version": 1, "data": prepared_data, "sam_weights": config.model.sam_weights, "sam_revision": config.model.sam_revision, "sam_commit": config.model.sam_commit, "coordinates": "z-up-aabb-unit-cube-surface64-v1", "visibility": "navi-positive-z-positive-image-y-v2"}
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
    order = np.random.default_rng(config.data.seed).permutation(objects).tolist()
    splits = {name: "train" if index < config.data.train_objects else "val" if index < config.data.train_objects + config.data.val_objects else "test" for index, name in enumerate(order)}
    atomic_json(output / "object_splits.json", splits)
    model = NativeSAM(config.model, config.data.frames, torch.device("cuda"))
    checkpoint = Path(model.checkpoint_root)
    fuser_config = OmegaConf.load(checkpoint / "ss_generator.yaml")["module"]["condition_embedder"]["backbone"]
    for embedder, _ in fuser_config["embedder_list"]:
        if "dino_model" in embedder:
            embedder["repo_or_dir"] = "/work/dependencies/dinov2"
            embedder["source"] = "local"
    fuser = load_component(fuser_config, checkpoint / "ss_generator.ckpt", "_base_models.condition_embedder.").cuda()
    fuser.force_drop_modalities = ["pointmap", "rgb_pointmap"]
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
            encoded = model.encode_geometry(occupancy)
            latent = encoded["shape"].transpose(1, 2).reshape(1, 8, 16, 16, 16)
            reconstruction = model.decoder(latent)
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
                tokens = fuser(**reference_inputs(Path(frames[0]), Path(masks[0])))
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
