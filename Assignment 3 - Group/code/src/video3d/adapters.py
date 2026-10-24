from __future__ import annotations

import math
import hashlib
import re
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, seed: int = 42) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        generator = torch.Generator(device="cpu" if base.weight.device.type == "meta" else base.weight.device).manual_seed(seed)
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5), generator=generator)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = torch.nn.functional.linear(torch.nn.functional.linear(x.float(), self.a), self.b)
        return self.base(x) + delta.to(x.dtype) * self.scale


def inject_lora(model: nn.Module, pattern: str, rank: int, alpha: float) -> list[str]:
    targets = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear) and re.fullmatch(pattern, name)]
    if not targets:
        raise ValueError(f"No linear modules match LoRA selector: {pattern}")
    for name, module in targets:
        parent, _, leaf = name.rpartition(".")
        stable_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
        model.get_submodule(parent).set_submodule(leaf, LoRALinear(module, rank, alpha, seed=stable_seed))
    return [name for name, _ in targets]


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if parameter.requires_grad}


def restore_trainable(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    expected = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if expected.keys() != state.keys():
        raise ValueError("Adapter checkpoint does not match enabled trainable parameters")
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(state[name].to(parameter.device))
