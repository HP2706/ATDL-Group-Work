"""Shape metrics on 64^3 voxel grids. Aligned metrics (ICP, Chamfer, F-score) arrive in P6."""

from __future__ import annotations

import torch


def voxel_iou(pred, gt) -> float:
    """Canonical voxel IoU (no alignment) of two boolean grids; 1.0 if both are empty."""
    pred = torch.as_tensor(pred).bool()
    gt = torch.as_tensor(gt).bool().to(pred.device)
    union = (pred | gt).sum().item()
    return 1.0 if union == 0 else (pred & gt).sum().item() / union
