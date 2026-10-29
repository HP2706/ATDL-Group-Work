"""Our foreground attentive pooler, using the upstream V-JEPA pooler implementation."""
from __future__ import annotations

import torch
from torch import nn
from video3d.config import AdapterConfig


class MaskPooler(nn.Module):
    def __init__(self, cfg: AdapterConfig) -> None:
        super().__init__()
        from src.models.attentive_pooler import AttentivePooler
        self.pooler = AttentivePooler(num_queries=cfg.pooler_queries,
                                     embed_dim=cfg.d_vid, num_heads=cfg.heads,
                                     depth=cfg.pooler_depth)
        self.projector = nn.Linear(cfg.d_vid, cfg.d_cond)
        self.threshold = cfg.foreground_threshold

    def condition(self, cond: dict[str, torch.Tensor], tokens: torch.Tensor,
                  coverage: torch.Tensor, query_sets: list[str],
                  padding: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        selected = coverage >= self.threshold
        if padding is not None:
            selected = selected & ~padding
        if not selected.any(dim=1).all():
            raise ValueError("No foreground tubelet passes the configured threshold")
        pooled = torch.cat([self.pooler(tokens[i:i + 1, mask])
                            for i, mask in enumerate(selected)], dim=0)
        projected = self.projector(pooled)
        output: dict[str, torch.Tensor] = {}
        for name, value in cond.items():
            if name == query_sets[0]:
                output[name] = projected
            elif name not in query_sets:
                output[name] = value
        return output

    def gates(self) -> dict[str, float]:
        return {}
