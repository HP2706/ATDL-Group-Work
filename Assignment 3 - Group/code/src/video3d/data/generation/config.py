"""Generator configuration, loaded from JSON and stored verbatim next to the dataset."""

from __future__ import annotations

from typing import Any, get_origin, get_type_hints
import json
import chz
from pathlib import Path



@chz.chz
class Config:
    width: int = 512
    height: int = 512
    hfov_deg: float = 50.0
    num_frames: int = 32

    n_objects_choices: list[int] = chz.field(default_factory=lambda: [1, 2, 3, 4])
    n_objects_probs: list[float] = chz.field(default_factory=lambda: [0.30, 0.30, 0.25, 0.15])
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

    # Opt-in target/occluder placement. Existing datasets retain random sampling.
    occlusion_mode: str = "random"
    occlusion_reference_range: tuple[float, float] = (0.30, 0.70)
    occlusion_min_frames: int = 8
    occlusion_frame_threshold: float = 0.20
    occlusion_min_best_visibility: float = 0.75
    occlusion_min_recovery: float = 0.30
    occlusion_surface_samples: int = 4096
    occluder_scale_ratio: float = 0.55
    occluder_scale_ratio_range: tuple[float, float] = (0.30, 1.30)
    render_devices: list[str] = chz.field(default_factory=lambda: ["OPTIX", "CUDA"])
    render_threads: int = 4

    # Geometric asset filter (paper App. B.1), applied to the train assets' filter_scores and
    # num_voxels; see video3d.data.generation.assetfilter. Chosen on the contact sheets of 100 random assets
    # (extent_min < 0.1 caught flat discs and specks; the rest of the thin objects were legitimate).
    # An asset is rejected if any of:
    #   extent_min < filter_extent_min              flat or point-like bounding box
    #   normal_eig_min < filter_normal_eig_min      sheet: all normals (anti)parallel; thin rods also
    #                                               score low (0.05 for a carrot), so keep this small
    #   outlier_area_frac > filter_outlier_area_frac  most of the surface sits far from the largest part;
    #                                               legit characters with detached limbs reach 0.7
    #   num_voxels < filter_min_voxels              near-zero volume; this is where stray far-away
    #                                               fragments show up, since normalization shrinks the
    #                                               real object (specks had 12-144 voxels, the next
    #                                               smallest asset 2169)
    filter_extent_min: float = 0.1
    filter_normal_eig_min: float = 0.03
    filter_outlier_area_frac: float = 0.9
    filter_min_voxels: int = 500
    # Objaverse also holds whole scenes (floor plans, houses on a ground plate, city blocks) that pass
    # the geometric criteria. An asset is rejected if a TRELLIS-500K caption contains one of these
    # whole words (case-insensitive); a toy house is an accepted false positive. "interior" is left
    # out because vehicle captions use it.
    filter_scene_words: list[str] = chz.field(default_factory=lambda: [
        "house", "houses", "two-story", "cottage", "cabin", "building", "buildings", "room", "rooms",
        "bedroom", "bathroom", "kitchen", "apartment", "floor plan", "floorplan", "architecture",
        "architectural", "city", "town", "village", "street", "landscape", "terrain", "island",
        "diorama", "environment",
    ])

    dataset_root: str = "dataset"
    bg_pool_dir: str = "data/bg_pools"
    blender_install_path: str = ".blender"
    # An existing Blender to use instead of BlenderProc's default 4.2.1, e.g. 4.5 LTS for GPUs
    # that 4.2 has no kernels for.
    blender_custom_path: str | None = None

    @property
    def min_visible_px(self) -> int:
        return int(round(self.min_visible_frac_of_image * self.width * self.height))

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        values: dict[str, Any] = json.loads(Path(path).read_text()) if path else {}
        hints = get_type_hints(cls)
        values = {key: tuple(value) if get_origin(hints.get(key)) is tuple else value for key,value in values.items()}
        return chz.Blueprint(cls).apply(values).make()

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(chz.asdict(self)))
