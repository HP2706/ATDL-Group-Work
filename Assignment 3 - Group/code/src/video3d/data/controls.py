"""Deterministic evaluation controls preserving the SAM reference inputs."""
from __future__ import annotations
from typing import Any

import numpy as np
import torch
from video3d.utils import sample_seed


def sample_id(info: dict[str, Any]) -> str:
    """Stable id of an eval sample."""
    return f"{info['clip_id']}/{info['k']}/{info['r']}"


def shuffle_frames(sample: dict[str, Any]) -> dict[str, Any]:
    """H3 control: permute the clip frames (seeded by the sample id); Δt slots stay in place."""
    x, info = sample["inputs"], sample["info"]
    K = x["frames"].shape[0]
    rng = np.random.default_rng(sample_seed(sample_id(info), "shuffle"))
    perm = torch.from_numpy(rng.permutation(K))
    ref_pos = int((perm == x["ref_pos"]).nonzero()[0, 0])
    inputs = {**x, "frames": x["frames"][perm], "frame_masks": x["frame_masks"][perm], "ref_pos": ref_pos}
    return {"inputs": inputs, "info": {**info, "frame_ids": [info["frame_ids"][i] for i in perm]}}


def wrong_video_partner(ds: Any, i: int) -> int:
    """T7: a deterministic sample from a different clip (about half the dataset away)."""
    clip_id = ds.index[i]["clip_id"]
    for step in range(len(ds)):
        j = (i + len(ds) // 2 + step) % len(ds)
        if ds.index[j]["clip_id"] != clip_id:
            return j
    raise AssertionError("wrong-video control needs at least two clips")


def with_video_of(sample: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """Keep the SAM 3D reference inputs; take the clip (frames, masks, Δt) from another sample."""
    keys = ("frames", "frame_masks", "dt", "ref_pos")
    inputs = {**sample["inputs"], **{k: other["inputs"][k] for k in keys}}
    return {"inputs": inputs, "info": {**sample["info"], "video_from": sample_id(other["info"])}}


def repeat_reference(sample: dict[str, Any]) -> dict[str, Any]:
    inputs = sample["inputs"]
    count = len(inputs["frames"])
    pos = inputs["ref_pos"]
    changed = {**inputs, "frames": inputs["frames"][pos:pos+1].expand(count, -1, -1, -1),
               "frame_masks": inputs["frame_masks"][pos:pos+1].expand(count, -1, -1),
               "dt": torch.zeros_like(inputs["dt"])}
    return {**sample, "inputs": changed}
