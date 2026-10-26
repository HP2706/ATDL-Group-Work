"""Generator configuration, loaded from YAML and stored verbatim next to the dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class Config:
    width: int = 512
    height: int = 512
    hfov_deg: float = 50.0
    num_frames: int = 32

    n_objects_choices: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    n_objects_probs: list[float] = field(default_factory=lambda: [0.30, 0.30, 0.25, 0.15])
    scale_range: tuple[float, float] = (0.4, 1.0)
    # Placement ball radius is placement_radius_base * N^(1/3), so the ball volume per object
    # stays constant and N = 4 objects at s = 1 still fit.
    placement_radius_base: float = 1.0
    max_placement_attempts: int = 20000

    # Applied to the tight scene radius (actual geometry, not bounding cubes). Below ~1.1 the
    # scene sphere may exceed the frame: at (1.0, 1.4), 3% of objects are truncated in some frame.
    orbit_factor_range: tuple[float, float] = (1.0, 1.4)
    start_elevation_range_deg: tuple[float, float] = (10.0, 45.0)
    arc_span_range_deg: tuple[float, float] = (10.0, 70.0)
    elevation_drift_range_deg: tuple[float, float] = (-10.0, 10.0)
    max_arc_retries: int = 50

    bg_hfov_deg: float = 130.0
    bg_margin_px: float = 2.0

    sun_strength_range: tuple[float, float] = (2.0, 5.0)
    ambient_strength_range: tuple[float, float] = (0.15, 0.35)

    cycles_samples: int = 64
    # BlenderProc's "INTEL" is a CPU compositor node (~1.2 s/frame); "OPTIX" runs inside Cycles on the GPU.
    denoiser: str = "OPTIX"

    min_visible_frac_of_image: float = 0.002
    min_visible_over_amodal: float = 0.10
    precheck_resolution: int = 128
    precheck_slack: float = 0.7
    max_retries: int = 20

    dataset_root: str = "dataset"
    bg_pool_dir: str = "data/bg_pools"
    blender_install_path: str = ".blender"

    @property
    def min_visible_px(self) -> int:
        return int(round(self.min_visible_frac_of_image * self.width * self.height))

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        data = yaml.safe_load(Path(path).read_text()) if path else {}
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        cfg = cls(**data)
        for name, f in cls.__dataclass_fields__.items():
            if "tuple" in str(f.type):
                setattr(cfg, name, tuple(getattr(cfg, name)))
        return cfg

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}
