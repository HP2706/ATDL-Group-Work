"""Shape metrics on 64^3 voxel grids; points are occupied voxel centres.

- Canonical vIoU: no alignment (tests the canonical orientation too).
- Aligned metrics (SAM 3D paper protocol): normalise both shapes to [-1, 1] (bbox centre, longest
  side), align prediction to GT with ICP (pytorch3d; similarity by default, because bbox
  normalisation of a rotated shape changes its scale), then vIoU (all voxels), and Chamfer and
  F-score on surface voxels. F-score thresholds are fractions of the longest side (0.02 ~ 1.3
  voxels at 64^3).
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig


def voxel_iou(pred, gt) -> float:
    """Canonical voxel IoU (no alignment) of two boolean grids; 1.0 if both are empty."""
    pred = torch.as_tensor(pred).bool()
    gt = torch.as_tensor(gt).bool().to(pred.device)
    union = (pred | gt).sum().item()
    return 1.0 if union == 0 else (pred & gt).sum().item() / union


def voxel_points(vox: torch.Tensor) -> torch.Tensor:
    """(M, 3) float centres of occupied voxels in the canonical [-0.5, 0.5]^3 cube."""
    return (torch.nonzero(vox).float() + 0.5) / vox.shape[0] - 0.5


def normalize_points(p: torch.Tensor) -> torch.Tensor:
    """Centre the bounding box and scale its longest side to [-1, 1]."""
    lo, hi = p.min(0).values, p.max(0).values
    return (p - (lo + hi) / 2) * (2.0 / (hi - lo).max().clamp_min(1e-8))


def subsample(p: torch.Tensor, n: int, seed: int = 0) -> torch.Tensor:
    """At most n points, chosen with a fixed seed."""
    if len(p) <= n:
        return p
    g = torch.Generator().manual_seed(seed)
    return p[torch.randperm(len(p), generator=g)[:n].to(p.device)]


def voxelize(p: torch.Tensor, res: int = 64) -> torch.Tensor:
    """Boolean res^3 grid of points in [-1, 1]^3."""
    idx = ((p + 1) / 2 * res).floor().long().clamp_(0, res - 1)
    grid = torch.zeros(res, res, res, dtype=torch.bool, device=p.device)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid


def surface(vox: torch.Tensor) -> torch.Tensor:
    """Occupied voxels with at least one empty 6-neighbour."""
    p = torch.nn.functional.pad(vox[None, None].float(), (1, 1, 1, 1, 1, 1))[0, 0]
    interior = vox.clone()
    for dim in range(3):
        for shift in (-1, 1):
            interior &= torch.roll(p, shift, dim)[1:-1, 1:-1, 1:-1].bool()
    return vox & ~interior


def icp_fit(src: torch.Tensor, tgt: torch.Tensor, iterations: int, estimate_scale: bool):
    """ICP (pytorch3d) aligning src (N, 3) to tgt (M, 3); returns the fitted map x -> s x R + T."""
    from pytorch3d.ops import iterative_closest_point
    sol = iterative_closest_point(src[None], tgt[None], max_iterations=iterations,
                                  estimate_scale=estimate_scale)
    R, T, scale = sol.RTs.R[0], sol.RTs.T[0], sol.RTs.s[0]
    return lambda x: scale * x @ R + T


def nn_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance from every point of a to its nearest neighbour in b."""
    from pytorch3d.ops import knn_points
    return knn_points(a[None], b[None], K=1).dists[0, :, 0].clamp_min(0).sqrt()


def shape_metrics(pred_vox, gt_vox, mcfg: DictConfig, seed: int = 0) -> dict:
    """All metrics of one prediction; the aligned ones are None if the prediction is empty."""
    pred_vox, gt_vox = torch.as_tensor(pred_vox).bool(), torch.as_tensor(gt_vox).bool()
    out = {"viou": voxel_iou(pred_vox, gt_vox), "empty_pred": not bool(pred_vox.any())}
    keys = ["viou_aligned", "chamfer"] + [f"fscore@{t}" for t in mcfg.fscore_thresholds]
    if out["empty_pred"]:
        return {**out, **{k: None for k in keys}}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = gt_vox.shape[0]

    def unit_frame(vox):                     # bbox centre and scale mapping the shape to [-1, 1]
        pts = voxel_points(vox)
        lo, hi = pts.min(0).values, pts.max(0).values
        return ((lo + hi) / 2).to(device), float(2.0 / (hi - lo).max().clamp_min(1e-8))

    (c_p, s_p), (c_g, s_g) = unit_frame(pred_vox), unit_frame(gt_vox)
    to_unit = lambda vox, c, sc: (voxel_points(vox).to(device) - c) * sc
    pred_surf = subsample(to_unit(surface(pred_vox), c_p, s_p), mcfg.max_points, seed)
    gt_surf = subsample(to_unit(surface(gt_vox), c_g, s_g), mcfg.max_points, seed)
    align = icp_fit(pred_surf, gt_surf, mcfg.icp_iterations, bool(mcfg.icp_scale))
    pred_surf = align(pred_surf)
    # aligned vIoU on the GT's own 64^3 grid: map the aligned prediction back into the GT frame
    pred_in_gt = align(to_unit(pred_vox, c_p, s_p)) / s_g + c_g
    out["viou_aligned"] = voxel_iou(voxelize((pred_in_gt * 2).clamp(-1, 1 - 1e-6), res), gt_vox.to(device))
    d_pg, d_gp = nn_distances(pred_surf, gt_surf), nn_distances(gt_surf, pred_surf)
    out["chamfer"] = float(d_pg.mean() + d_gp.mean())
    for t in mcfg.fscore_thresholds:
        tau = 2.0 * t                                          # longest side spans 2 in [-1, 1]
        precision, recall = float((d_pg < tau).float().mean()), float((d_gp < tau).float().mean())
        out[f"fscore@{t}"] = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return out
