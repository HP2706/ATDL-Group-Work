"""Trainable-state handling, including LoRA and EMA, without copying frozen models."""
from __future__ import annotations
from contextlib import contextmanager
from collections.abc import Iterator
import torch
from torch import nn


def extra_parameters(encoder: nn.Module, sam: nn.Module) -> dict[str, nn.Parameter]:
    return {f"{prefix}.{name}": p for prefix, module in [("encoder",encoder),("sam",sam)]
            for name,p in module.named_parameters() if p.requires_grad}


def restore_extra(parameters: dict[str, nn.Parameter], values: dict[str, torch.Tensor]) -> None:
    if parameters.keys() != values.keys():
        raise ValueError("Checkpoint LoRA parameter names do not match the selected model")
    with torch.no_grad():
        for name,p in parameters.items():
            p.copy_(values[name])


@contextmanager
def use_ema(parameters: dict[str, nn.Parameter], ema: dict[str, torch.Tensor]) -> Iterator[None]:
    previous = {name:p.detach().clone() for name,p in parameters.items()}
    restore_extra(parameters,ema)
    try:
        yield
    finally:
        restore_extra(parameters,previous)
