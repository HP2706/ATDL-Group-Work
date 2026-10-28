"""Dataset housekeeping: `gen-data` (one shard of the external generator), `merge-manifests`,
`validate-data` (file presence, shapes, dtypes, guarantees, data-format §8 projection check).

The generator is maintained by Julian; its command template in `generator.command` is a
placeholder to confirm with him. Shard manifests are written here from each clip's meta.json,
so they do not depend on the generator's output naming.
"""

from __future__ import annotations

import json
import logging
import math
import shlex
import subprocess
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from PIL import Image

from vidsam3d.data import reader
from vidsam3d.utils import setup_logging

log = logging.getLogger("vidsam3d")
MIN_VISIBLE_FRAC = 0.1          # data format §6: >= 10 % of the amodal area visible
MIN_VISIBLE_IMAGE_FRAC = 0.002  # data format §6: >= 0.2 % of the image (524 px at 512^2)


def shard_seeds(cfg: DictConfig, split: str, shard_id: int, num_shards: int) -> tuple[int, int]:
    """(first seed, number of clips) of one shard; splits use disjoint seed ranges."""
    total = int(cfg.generator.num_clips[split])
    assert 0 <= shard_id < num_shards, f"shard {shard_id} not in [0, {num_shards})"
    per = math.ceil(total / num_shards)
    first = shard_id * per
    return int(cfg.generator.seed_offset[split]) + first, max(0, min(per, total - first))


def shard_manifest_path(data_root: Path, split: str, shard_id: int) -> Path:
    """dataset/manifest.<split>.shard<ID>.jsonl"""
    return data_root / f"manifest.{split}.shard{shard_id:03d}.jsonl"


def manifest_line(meta: dict, split: str) -> dict:
    """Manifest entry of one clip, from its meta.json."""
    return {"clip_id": meta["clip_id"], "split": split, "N": meta["num_objects"],
            "arc_span_deg": meta["arc_span_deg"], "seed": meta["seed"],
            "path": f"clips/{split}/{meta['clip_id']}"}


def gen_data(cfg: DictConfig, split: str, shard_id: int, num_shards: int, dry_run: bool = False) -> Path:
    """Run one shard of the external generator, then write its shard manifest."""
    setup_logging()
    data_root = Path(cfg.paths.data_root)
    seed_start, num_clips = shard_seeds(cfg, split, shard_id, num_shards)
    values = {"generator_repo": cfg.generator.repo, "split": split, "seed_start": seed_start,
              "num_clips": num_clips, "out_dir": data_root, "shard_id": shard_id}
    cmd = cfg.generator.command.format(**{k: shlex.quote(str(v)) for k, v in values.items()})
    log.info("shard %d/%d of %s: seeds [%d, %d)", shard_id, num_shards, split, seed_start,
             seed_start + num_clips)
    log.info("$ %s", cmd)
    if dry_run:
        return shard_manifest_path(data_root, split, shard_id)
    subprocess.run(shlex.split(cmd), check=True)
    seeds = range(seed_start, seed_start + num_clips)
    metas = [json.loads(p.read_text()) for p in sorted((data_root / "clips" / split).glob("*/meta.json"))]
    lines = [manifest_line(m, split) for m in metas if m["seed"] in seeds]
    if len(lines) != num_clips:
        log.warning("expected %d clips for this shard, found %d", num_clips, len(lines))
    path = shard_manifest_path(data_root, split, shard_id)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    log.info("wrote %s (%d clips)", path, len(lines))
    return path


def merge_manifests(cfg: DictConfig, split: str) -> Path:
    """Replace the split's lines in manifest.jsonl by its shard manifests; refuse duplicates."""
    setup_logging()
    data_root = Path(cfg.paths.data_root)
    shards = sorted(data_root.glob(f"manifest.{split}.shard*.jsonl"))
    assert shards, f"no manifest.{split}.shard*.jsonl in {data_root}"
    keep = [e for e in (reader.read_manifest(data_root) if (data_root / "manifest.jsonl").is_file() else [])
            if e["split"] != split]
    new = [json.loads(line) for p in shards for line in p.read_text().splitlines() if line.strip()]
    ids = [e["clip_id"] for e in keep + new]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    assert not dupes, f"duplicate clip_ids: {dupes[:10]}"
    out = data_root / "manifest.jsonl"
    tmp = data_root / "manifest.jsonl.tmp"
    tmp.write_text("".join(json.dumps(e) + "\n" for e in keep + new))
    tmp.replace(out)
    log.info("%s: %d clips of %s from %d shards (%d other clips kept)", out, len(new), split,
             len(shards), len(keep))
    return out


def check_clip(clip: reader.Clip, entry: dict, frames: list[int], voxels_seen: set) -> list[str]:
    """Problems found in one clip (empty list if it is fine)."""
    errors, d = [], clip.dir
    T, N, (H, W) = clip.num_frames, clip.num_objects, clip.image_size
    if N != entry["N"] or clip.meta["num_objects"] != N:
        errors.append(f"N mismatch: manifest {entry['N']}, objects.json {N}, meta {clip.meta['num_objects']}")
    expected = [f"{sub}/{t:03d}.png" for sub in ("rgb", "rgba_fg", "mask_visible") for t in range(T)]
    expected += [f"depth/{t:03d}.npz" for t in range(T)]
    expected += [f"mask_amodal/obj{k}/{t:03d}.png" for k in range(1, N + 1) for t in range(T)]
    missing = [f for f in expected if not (d / f).is_file()]
    if missing:
        return errors + [f"{len(missing)} missing files, e.g. {missing[0]}"]
    for k in range(1, N + 1):
        o = reader.object_entry(clip, k)
        if o["voxels"] not in voxels_seen:
            vox = np.load(clip.root / o["voxels"])
            if vox.dtype != bool or vox.shape != (64, 64, 64):
                errors.append(f"{o['voxels']}: {vox.dtype} {vox.shape}, expected bool (64, 64, 64)")
            voxels_seen.add(o["voxels"])
    vis = reader.visibility(clip)
    if vis.shape != (T, N):
        errors.append(f"visibility shape {vis.shape}, expected {(T, N)}")
    elif vis.min() < MIN_VISIBLE_FRAC:
        errors.append(f"visible_frac {vis.min():.3f} < {MIN_VISIBLE_FRAC}")
    elif reader.visibility(clip, "visible_px").min() < MIN_VISIBLE_IMAGE_FRAC * H * W:
        errors.append(f"visible_px below {MIN_VISIBLE_IMAGE_FRAC:.1%} of the image")
    for t in frames:
        rgb, inst = reader.load_rgb(clip, t), reader.load_instance_mask(clip, t)
        depth = reader.load_depth_gt(clip, t)
        if rgb.shape != (H, W, 3) or rgb.dtype != np.uint8:
            errors.append(f"rgb/{t:03d}: {rgb.dtype} {rgb.shape}")
        if inst.dtype != np.uint8 or inst.max() > N:
            errors.append(f"mask_visible/{t:03d}: {inst.dtype}, max {inst.max()} > N={N}")
        if depth.dtype != np.float32 or depth.shape != (H, W) or not np.array_equal(np.isfinite(depth), inst > 0):
            errors.append(f"depth/{t:03d}: {depth.dtype} {depth.shape} or not finite exactly on objects")
        for k in range(1, N + 1):
            amodal = np.asarray(Image.open(d / "mask_amodal" / f"obj{k}" / f"{t:03d}.png"))
            if not set(np.unique(amodal)) <= {0, 255} or ((inst == k) & (amodal == 0)).any():
                errors.append(f"mask_amodal/obj{k}/{t:03d}: not {{0, 255}} or visible not inside amodal")
    return errors


def validate_data(cfg: DictConfig, split: str, num_projection_clips: int = 20) -> dict:
    """Check every clip of a split; run the voxel-projection check on a seeded random subset."""
    setup_logging()
    data_root = Path(cfg.paths.data_root)
    entries = reader.read_manifest(data_root, split)
    assert entries, f"no clips of split {split!r} in {data_root}/manifest.jsonl"
    rng = np.random.default_rng(int(cfg.seed))
    projection_ids = set(rng.choice([e["clip_id"] for e in entries],
                                    size=min(num_projection_clips, len(entries)), replace=False))
    problems, voxels_seen, projection = {}, set(), []
    for e in entries:
        try:
            clip = reader.load_clip(data_root, e)
            T = clip.num_frames
            errors = check_clip(clip, e, sorted({0, T // 2, T - 1}), voxels_seen)
            if not errors and e["clip_id"] in projection_ids:
                for k in range(1, clip.num_objects + 1):
                    frac = reader.voxel_projection_check(clip, k, T // 2)
                    projection.append(frac)
                    if frac <= 0.9:
                        errors.append(f"object {k}: only {frac:.2f} of voxel centres on the amodal mask")
        except Exception as ex:                   # unreadable JSON / image / npz
            errors = [f"{type(ex).__name__}: {ex}"]
        if errors:
            problems[e["clip_id"]] = errors
    report = {"split": split, "clips": len(entries), "clips_with_problems": len(problems),
              "projection_checked_objects": len(projection),
              "projection_min": float(min(projection)) if projection else None, "problems": problems}
    for clip_id, errors in list(problems.items())[:20]:
        log.error("%s: %s", clip_id, "; ".join(errors))
    log.info("validate-data %s: %d clips, %d with problems, projection check min %s",
             split, len(entries), len(problems), report["projection_min"])
    return report
