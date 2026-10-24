"""Scene layout, camera trajectory and lighting sampling (spec Sections 4.3, 4.4, 4.5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .conventions import look_at_opencv, random_rotation

HALF_DIAGONAL = np.sqrt(3) / 2


class PlacementError(RuntimeError):
    pass


@dataclass
class ObjectPose:
    asset_id: str
    R_wo: np.ndarray
    t_wo: np.ndarray
    s_wo: np.ndarray

    @property
    def bounding_radius(self) -> float:
        return float(self.s_wo[0] * HALF_DIAGONAL)


@dataclass
class Trajectory:
    center: np.ndarray
    orbit_radius: float
    start_azimuth_deg: float
    start_elevation_deg: float
    arc_span_deg: float
    elevation_drift_deg: float
    num_frames: int

    def eased(self) -> np.ndarray:
        t = np.arange(self.num_frames)
        return (1 - np.cos(np.pi * t / (self.num_frames - 1))) / 2

    def camera_at(self, u: float) -> np.ndarray:
        """T_cw at eased arc parameter u in [0, 1]."""
        az = np.radians(self.start_azimuth_deg + self.arc_span_deg * u)
        el = np.radians(self.start_elevation_deg + self.elevation_drift_deg * u)
        eye = self.center + self.orbit_radius * np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
        )
        return look_at_opencv(eye, self.center)

    def frames(self) -> list[np.ndarray]:
        return [self.camera_at(u) for u in self.eased()]

    def reference(self) -> np.ndarray:
        return self.camera_at(0.5)


def sample_num_objects(rng: np.random.Generator, cfg: Config) -> int:
    return int(rng.choice(cfg.n_objects_choices, p=cfg.n_objects_probs))


def placement_radius(cfg: Config, n: int) -> float:
    return cfg.placement_radius_base * n ** (1 / 3)


def sample_in_ball(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    d = rng.standard_normal((n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d * radius * rng.random((n, 1)) ** (1 / 3)


def sample_layout(rng: np.random.Generator, cfg: Config, asset_ids: list[str]) -> list[ObjectPose]:
    """Rotations and scales once, then positions resampled until bounding spheres are disjoint."""
    n = len(asset_ids)
    rotations = [random_rotation(rng) for _ in range(n)]
    scales = rng.uniform(*cfg.scale_range, size=n)
    radii = scales * HALF_DIAGONAL
    min_dist = radii[:, None] + radii[None, :]
    iu = np.triu_indices(n, 1)
    ball = placement_radius(cfg, n)
    for _ in range(cfg.max_placement_attempts):
        t = sample_in_ball(rng, n, ball)
        dist = np.linalg.norm(t[:, None] - t[None, :], axis=-1)
        if np.all(dist[iu] > min_dist[iu]):
            break
    else:
        raise PlacementError(f"no overlap-free placement for N={n} in {cfg.max_placement_attempts} tries")
    return [
        ObjectPose(a, R, p, np.full(3, s)) for a, R, p, s in zip(asset_ids, rotations, t, scales)
    ]


def scene_extent(objects: list[ObjectPose]) -> tuple[np.ndarray, float]:
    """Centroid c = mean(t_wo) and radius r = max_k(|t_k - c| + bounding radius_k)."""
    c = np.mean([o.t_wo for o in objects], axis=0)
    r = max(np.linalg.norm(o.t_wo - c) + o.bounding_radius for o in objects)
    return c, float(r)


def sample_trajectory(rng: np.random.Generator, cfg: Config, center: np.ndarray, radius: float) -> Trajectory:
    factor = rng.uniform(*cfg.orbit_factor_range)
    traj = Trajectory(
        center=center,
        orbit_radius=float(radius / np.tan(np.radians(cfg.hfov_deg) / 2) * factor),
        start_azimuth_deg=float(rng.uniform(0.0, 360.0)),
        start_elevation_deg=float(rng.uniform(*cfg.start_elevation_range_deg)),
        arc_span_deg=0.0,
        elevation_drift_deg=0.0,
        num_frames=cfg.num_frames,
    )
    resample_arc(rng, cfg, traj)
    return traj


def resample_arc(rng: np.random.Generator, cfg: Config, traj: Trajectory) -> None:
    span = rng.uniform(*cfg.arc_span_range_deg)
    traj.arc_span_deg = float(span if rng.random() < 0.5 else -span)
    traj.elevation_drift_deg = float(rng.uniform(*cfg.elevation_drift_range_deg))


def sample_light(rng: np.random.Generator, cfg: Config) -> dict:
    """Sun direction (pointing *towards* the sun) uniform on the upper hemisphere, plus strengths."""
    d = rng.standard_normal(3)
    d[2] = abs(d[2])
    d /= np.linalg.norm(d)
    return {
        "sun_direction": d.tolist(),
        "sun_strength": float(rng.uniform(*cfg.sun_strength_range)),
        "ambient_strength": float(rng.uniform(*cfg.ambient_strength_range)),
    }
