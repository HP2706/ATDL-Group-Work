"""Build samples (clip_id, k, r, window): frame subset, reference frame, clip window, crop.

Names: k = target object id (1..N), r = reference frame, K = clip length (`clip_len`).
A sample is {"inputs": model inputs, "info": ids + stratification fields}. Model inputs never
contain amodal masks, depth, poses or cameras (`check_no_leakage`).
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import Dataset

from vidsam3d.data import reader

ALLOWED_INPUT_KEYS = frozenset({"frames", "frame_masks", "dt", "ref_pos", "ref_rgb", "ref_mask"})
FORBIDDEN_WORDS = ("amodal", "depth", "pose", "camera", "intrinsic", "extrinsic", "rotation",
                   "translation", "voxel", "latent")
TUBELET = 2  # frames per V-JEPA tubelet; Δt is counted in tubelets


def check_no_leakage(inputs: dict) -> None:
    """Assert that a model-input dict holds only allowed keys (no ground truth)."""
    keys = set(inputs)
    assert keys <= ALLOWED_INPUT_KEYS, f"unexpected model inputs: {sorted(keys - ALLOWED_INPUT_KEYS)}"
    bad = [key for key in keys if any(word in key.lower() for word in FORBIDDEN_WORDS)]
    assert not bad, f"ground-truth-like keys in model inputs: {bad}"


# ---- frames, reference, window ----

def frame_subset(offset: int, num: int = 16, stride: int = 2) -> list[int]:
    """S = {o, o + stride, ..., o + (num - 1) * stride}."""
    return [offset + stride * i for i in range(num)]


def valid_references(vis_frac_k: np.ndarray, subset: list[int], min_frac: float) -> list[int]:
    """Frames of S where object k has at least `min_frac` of its amodal area visible."""
    valid = [t for t in subset if vis_frac_k[t] >= min_frac]
    assert valid, "object is never sufficiently visible in S"
    return valid


def sample_reference(candidates: list[int], vis_frac_k: np.ndarray, rng: np.random.Generator,
                     p_occluded_bias: float) -> int:
    """Train: uniform over candidates; with prob. p_occluded_bias, ∝ (1 - visible_frac)."""
    if rng.random() < p_occluded_bias:
        w = 1.0 - np.array([vis_frac_k[t] for t in candidates])
        if w.sum() > 0:
            return int(rng.choice(candidates, p=w / w.sum()))
    return int(rng.choice(candidates))


def eval_references(candidates: list[int], vis_frac_k: np.ndarray,
                    middle: int) -> list[tuple[int, list[str]]]:
    """Eval: the middle frame and argmin visible_frac over S; one sample if they coincide."""
    occluded = min(candidates, key=lambda t: vis_frac_k[t])  # ties: earliest frame
    if middle == occluded:
        return [(middle, ["middle", "most_occluded"])]
    refs = [(occluded, ["most_occluded"])]
    if middle in candidates:
        refs.insert(0, (middle, ["middle"]))
    return refs


def sample_clip_length(rng: np.random.Generator, lengths: list[int], probs: list[float]) -> int:
    """Draw K from `lengths` with probabilities `probs`."""
    return int(rng.choice(lengths, p=np.asarray(probs) / np.sum(probs)))


def window_positions(n: int, ref_pos: int, clip_len: int) -> range:
    """Contiguous positions of S of length K containing ref_pos, centred where possible."""
    assert 1 <= clip_len <= n, f"K={clip_len} not in [1, {n}]"
    start = min(max(ref_pos - clip_len // 2, 0), n - clip_len)
    return range(start, start + clip_len)


def tubelet_offsets(clip_len: int, ref_in_window: int, tubelet: int = TUBELET) -> np.ndarray:
    """Δt of every tubelet relative to the tubelet containing r (K = 1: one image token set)."""
    if clip_len == 1:
        return np.zeros(1, dtype=np.int64)
    assert clip_len % tubelet == 0, f"K={clip_len} not divisible by tubelet {tubelet}"
    return np.arange(clip_len // tubelet, dtype=np.int64) - ref_in_window // tubelet


# ---- object-centred crop ----

def union_bbox(masks: list[np.ndarray]) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the union of boolean masks; x1, y1 exclusive."""
    ys, xs = np.nonzero(np.logical_or.reduce(masks))
    assert len(xs), "empty visible masks: cannot place a crop"
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_window(bbox: tuple[int, int, int, int], image_size: tuple[int, int], scale: float,
                min_side: int, rng: np.random.Generator | None = None, jitter_center: float = 0.0,
                jitter_scale: tuple[float, float] = (1.0, 1.0)) -> tuple[int, int, int]:
    """Square window (x0, y0, side) around the bbox; side clamped to the image size."""
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(scale * max(x1 - x0, y1 - y0), min_side)
    if rng is not None:  # train jitter, identical for all frames
        side *= rng.uniform(*jitter_scale)
        cx += rng.uniform(-jitter_center, jitter_center) * side
        cy += rng.uniform(-jitter_center, jitter_center) * side
    side = int(round(min(side, max(image_size))))
    return int(round(cx - side / 2)), int(round(cy - side / 2)), side


def crop_pad(img: np.ndarray, window: tuple[int, int, int]) -> np.ndarray:
    """Cut the square window out of an (H, W, ...) array, zero-padding outside the image."""
    x0, y0, s = window
    H, W = img.shape[:2]
    out = np.zeros((s, s) + img.shape[2:], img.dtype)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + s, W), min(y0 + s, H)
    if xa < xb and ya < yb:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = img[ya:yb, xa:xb]
    return out


def crop_frames(rgbs: list[np.ndarray], masks: list[np.ndarray], window: tuple[int, int, int],
                size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop + resize: RGB bilinear -> (K, 3, size, size) in [0, 1]; masks area -> (K, size, size)."""
    rgb = torch.from_numpy(np.stack([crop_pad(x, window) for x in rgbs]))
    rgb = rgb.permute(0, 3, 1, 2).float() / 255.0
    rgb = F.interpolate(rgb, size=(size, size), mode="bilinear", align_corners=False,
                        antialias=True).clamp_(0, 1)
    m = torch.from_numpy(np.stack([crop_pad(x, window) for x in masks]).astype(np.float32))
    m = F.interpolate(m[:, None], size=(size, size), mode="area")[:, 0]
    return rgb, m


# ---- one sample ----

def build_sample(clip: reader.Clip, k: int, r: int, offset: int, clip_len: int,
                 data_cfg: DictConfig, rng: np.random.Generator | None = None,
                 reference_rules: list[str] | None = None) -> dict:
    """Build one sample; `rng` given = train (crop jitter), None = deterministic eval."""
    subset = frame_subset(offset, data_cfg.num_subset_frames, data_cfg.frame_stride)
    assert subset[-1] < clip.num_frames, f"{clip.clip_id}: S={subset} exceeds {clip.num_frames} frames"
    ref_pos = subset.index(r)
    positions = window_positions(len(subset), ref_pos, clip_len)
    frame_ids = [subset[i] for i in positions]
    ref_in_window = ref_pos - positions.start

    rgbs = [reader.load_rgb(clip, t) for t in frame_ids]
    masks = [reader.load_visible_mask(clip, k, t) for t in frame_ids]
    c = data_cfg.crop
    window = crop_window(union_bbox(masks), clip.image_size, c.scale, c.min_side, rng,
                         c.jitter_center, tuple(c.jitter_scale))
    frames, frame_masks = crop_frames(rgbs, masks, window, c.size)

    inputs = {
        "frames": frames,                                    # (K, 3, S, S) in [0, 1]
        "frame_masks": frame_masks,                          # (K, S, S) fractional visible mask
        "dt": torch.from_numpy(tubelet_offsets(clip_len, ref_in_window)),
        "ref_pos": ref_in_window,                            # index of r in the window
        "ref_rgb": torch.from_numpy(rgbs[ref_in_window].copy()),   # (H, W, 3) uint8, full res
        "ref_mask": torch.from_numpy(masks[ref_in_window]),        # (H, W) bool, full res
    }
    check_no_leakage(inputs)
    info = {
        "clip_id": clip.clip_id, "k": k, "r": r, "clip_len": clip_len, "offset": offset,
        "frame_ids": frame_ids, "crop_window": window, "reference_rules": reference_rules or [],
        "asset_id": reader.object_entry(clip, k)["asset_id"],
        "visible_frac": float(reader.visibility(clip, "visible_frac")[r, k - 1]),
        "amodal_px": int(reader.visibility(clip, "amodal_px")[r, k - 1]),
        "arc_span_deg": reader.arc_span_deg(clip), "num_objects": clip.num_objects,
    }
    return {"inputs": inputs, "info": info}


# ---- splits ----

VAL = "val"


def is_val_asset(asset_id: str, val_asset_frac: float) -> bool:
    """Deterministic, seed-independent choice of the held-out validation assets."""
    return zlib.crc32(f"val:{asset_id}".encode()) / 2**32 < val_asset_frac


def split_folder(data_cfg: DictConfig, split: str) -> str:
    """Export folder of a split: val is carved out of the train folder."""
    return data_cfg.train_split if split == VAL else split


def split_objects(clip_assets: dict[str, tuple], split: str, data_cfg: DictConfig) -> dict[str, list[int]]:
    """{clip_id: target objects k} of a split. Val is asset-disjoint from train: a clip that shows
    a val asset belongs to val (with only the val assets as targets), all other clips to train."""
    frac = float(data_cfg.val_asset_frac)
    is_val = {cid: [is_val_asset(a, frac) for a in assets] for cid, assets in clip_assets.items()}
    if split == VAL:
        return {cid: [k for k, v in enumerate(flags, 1) if v] for cid, flags in is_val.items() if any(flags)}
    if split == data_cfg.train_split:
        return {cid: list(range(1, len(flags) + 1)) for cid, flags in is_val.items() if not any(flags)}
    return {cid: list(range(1, len(flags) + 1)) for cid, flags in is_val.items()}


# ---- dataset ----

class ClipDataset(Dataset):
    """Train: one item per (clip, k), with o, r, K and crop jitter drawn per item.
    Eval: one item per (clip, k, r) for the middle and most-occluded reference frames.
    `split` is train, val (asset-disjoint part of the train folder) or an export folder (eval)."""

    def __init__(self, cfg: DictConfig, split: str, train: bool, clip_len: int | None = None):
        self.data_cfg = cfg.data
        self.train = train
        self.seed = int(cfg.seed)
        self.epoch = 0
        self.clip_len = int(clip_len or cfg.eval.k)
        self.split = reader.open_split(Path(cfg.paths.data_root) / split_folder(cfg.data, split),
                                       str(cfg.paths.cache_root))
        clip_assets = {cid: ref.asset_ids for cid, ref in self.split.clips.items()}
        self.objects = split_objects(clip_assets, split, cfg.data)
        assert self.objects, f"no clips for split {split!r} in {self.split.dir}"
        if train:
            self.index = [{"clip_id": cid, "k": k} for cid, ks in self.objects.items() for k in ks]
        else:
            self.index = self._eval_index(int(cfg.eval.middle_frame))

    @property
    def voxels_file(self) -> Path:
        return self.split.voxels_file

    def clip(self, clip_id: str) -> reader.Clip:
        return self.split.load_clip(clip_id)

    def _eval_index(self, middle: int) -> list[dict]:
        """All (clip_id, k, r, rules) for the eval reference rules (o = 0)."""
        d = self.data_cfg
        subset = frame_subset(0, d.num_subset_frames, d.frame_stride)
        index = []
        for cid, ks in self.objects.items():
            vis = reader.visibility(self.clip(cid))
            for k in ks:
                cands = valid_references(vis[:, k - 1], subset, d.min_visible_amodal_frac)
                for r, rules in eval_references(cands, vis[:, k - 1], middle):
                    index.append({"clip_id": cid, "k": k, "r": r, "rules": rules})
        return index

    def set_epoch(self, epoch: int) -> None:
        """Change the per-item random draws (train)."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int | tuple[int, int]) -> dict:
        """Item i; train samplers may pass (i, epoch) so the draw depends only on the position."""
        i, epoch = i if isinstance(i, tuple) else (i, self.epoch)
        item = self.index[i]
        clip = self.clip(item["clip_id"])
        k, d = item["k"], self.data_cfg
        if not self.train:
            return build_sample(clip, k, item["r"], 0, self.clip_len, d,
                                reference_rules=item["rules"])
        rng = np.random.default_rng([self.seed, epoch, i])
        offset = int(rng.integers(0, 2))
        subset = frame_subset(offset, d.num_subset_frames, d.frame_stride)
        vis_k = reader.visibility(clip)[:, k - 1]
        cands = valid_references(vis_k, subset, d.min_visible_amodal_frac)
        r = sample_reference(cands, vis_k, rng, d.p_occluded_bias)
        clip_len = sample_clip_length(rng, list(d.clip_lengths), list(d.clip_length_probs))
        return build_sample(clip, k, r, offset, clip_len, d, rng)
