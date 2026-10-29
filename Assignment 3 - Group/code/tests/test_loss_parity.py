"""Quick check against the installed, pinned upstream loss, not a second copied formula."""
from __future__ import annotations

import os
from typing import Any
import pytest
import torch
from video3d.loss import FlowMatchingLoss

os.environ["LIDRA_SKIP_INIT"] = "1"


@pytest.mark.parametrize("sigma", [0.0, 0.2])
@pytest.mark.parametrize("time_scale", [1.0, 1000.0])
def test_loss_and_conditioning_gradients_match_upstream(sigma: float, time_scale: float) -> None:
    from sam3d_objects.model.backbone.generator.flow_matching.model import FlowMatching
    from sam3d_objects.model.backbone.generator.shortcut.model import ShortCut
    torch.manual_seed(7)
    truth, noise = torch.randn(2, 5, 3), torch.randn(2, 5, 3)
    times = torch.tensor([0.1, 0.9])

    def velocity(x: dict[str, torch.Tensor], t: torch.Tensor,
                 c: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        return {"shape": x["shape"] * c[:, None, :] + t[:, None, None] * 0.001}

    for reference_class in [FlowMatching, ShortCut]:
        kwargs = {"self_consistency_prob": 0.0} if reference_class is ShortCut else {}
        reference = reference_class(reverse_fn=velocity, sigma_min=sigma,
                                    time_scale=time_scale, loss_weights={"shape":1.0}, **kwargs)
        # Explicitly control the stochastic inputs, leaving upstream interpolation,
        # targets, invocation, weighting and reduction intact.
        reference._generate_t = lambda x: times
        reference._generate_x0 = lambda x: {"shape":noise}
        a = torch.randn(2,3,requires_grad=True)
        b = a.detach().clone().requires_grad_(True)
        expected, _ = reference.loss({"shape":truth}, a)
        actual, _ = FlowMatchingLoss(velocity, noise, times, time_scale, sigma).loss({"shape":truth}, b)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        grad_a, = torch.autograd.grad(expected,a)
        grad_b, = torch.autograd.grad(actual,b)
        torch.testing.assert_close(grad_b, grad_a, rtol=1e-6, atol=1e-7)


def test_time_sampler_matches_upstream_distribution() -> None:
    from sam3d_objects.model.backbone.generator.flow_matching.model import lognorm_sampler
    from video3d.config import LossConfig
    from video3d.train import sample_tau
    a, b = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    expected = lognorm_sampler(size=(100,), generator=a)
    actual = sample_tau(100,LossConfig(),b,torch.device("cpu"))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
