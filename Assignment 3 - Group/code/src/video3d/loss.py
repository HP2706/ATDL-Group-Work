# Copyright (c) Meta Platforms, Inc. and affiliates.
"""SAM 3D flow-matching loss, with explicit noise/time inputs for reproducible checks."""
from __future__ import annotations

from typing import Any, Callable
import optree
import torch
from torch.nn import functional as F


class FlowMatchingLoss:
    """Shape-only context for the upstream loss; the trainer owns random sampling."""

    def __init__(self, reverse_fn: Callable[..., dict[str, torch.Tensor]],
                 noise: torch.Tensor, time: torch.Tensor, time_scale: float = 1000.0,
                 sigma_min: float = 0.0) -> None:
        self.reverse_fn = reverse_fn
        self.noise, self.time = noise, time
        self.time_scale, self.sigma_min = time_scale, sigma_min
        self.loss_fn = F.mse_loss
        self.loss_weights = {"shape": 1.0}

    def _generate_t(self, x1: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.time

    def _generate_x0(self, x1: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"shape": self.noise}

    def _generate_xt(self, x0: dict[str, torch.Tensor], x1: dict[str, torch.Tensor],
                     t: torch.Tensor) -> dict[str, torch.Tensor]:
        def interpolate(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            tb = t.to(b.device).reshape(-1, *([1] * (b.ndim - 1)))
            return (1 - (1 - self.sigma_min) * tb) * a + tb * b
        return optree.tree_map(interpolate, x0, x1)

    def _generate_target(self, x0: dict[str, torch.Tensor],
                         x1: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return optree.tree_map(lambda a, b: b - (1 - self.sigma_min) * a, x0, x1)

    def loss(self, x1: dict[str, torch.Tensor], *args_conditionals: Any,
             **kwargs_conditionals: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        t = self._generate_t(x1)
        x0 = self._generate_x0(x1)
        x_t = self._generate_xt(x0, x1, t)
        target = self._generate_target(x0, x1)

        prediction = self.reverse_fn(
            x_t,
            t * self.time_scale,
            *args_conditionals,
            **kwargs_conditionals,
        )

        # broadcast & and compute loss
        loss = optree.tree_broadcast_map(
            lambda fn, weight, pred, targ: weight * fn(pred, targ),
            self.loss_fn,
            self.loss_weights,
            prediction,
            target,
        )

        total_loss = sum(optree.tree_flatten(loss)[0])

        # Create detailed loss breakdown
        detail_losses = {
            "flow_matching_loss": total_loss,
        }
        if isinstance(loss, dict):
            detail_losses.update(loss)
        return total_loss, detail_losses

    # Copied from FlowMatching.loss in sam3d_objects/model/backbone/generator/
    # flow_matching/model.py, commit f91db411c50efee93d8db7aeb323885650f6f722.
    # Adaptations: annotations; noise/time supplied explicitly through this context;
    # shape-only weights. The loss body and reduction are unchanged.


def flow_loss(sam3d: Any, cond: dict[str, torch.Tensor], x1: torch.Tensor,
              tau: torch.Tensor, x0: torch.Tensor,
              layout: dict[str, torch.Tensor]) -> torch.Tensor:
    """Single production loss path for both conditioning strategies."""
    def velocity(x: dict[str, torch.Tensor], time: torch.Tensor,
                 conditioning: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        prediction, _ = sam3d.velocity(x["shape"], time / 1000.0, conditioning, layout)
        return {"shape": prediction.float()}
    loss, _ = FlowMatchingLoss(velocity, x0, tau).loss({"shape": x1}, cond)
    return loss
