"""T1, T3, T4 on every clip listed in the dataset manifest."""

import warnings

import numpy as np
import pytest

from sam3dvid.background import covers, homography, relative_rotation, warp
from sam3dvid.validate import (
    load_canonical_mesh,
    projection_consistency,
    raycast_consistency,
    sample_surface,
    visibility_failures,
    visibility_stats,
)

# Calibrated on 576 correct object-frames (max 0.007 and 1e-5) against perturbed poses
# (a half-pixel principal point shift gives at least 0.129 boundary mismatch).
T1B_MAX_BOUNDARY_MISMATCH = 0.05
T1B_MAX_REL_DEPTH_ERR = 5e-4


def t1b_fails(m: dict) -> bool:
    return m["boundary_mismatch"] > T1B_MAX_BOUNDARY_MISMATCH or not m["rel_depth_err_median"] <= T1B_MAX_REL_DEPTH_ERR


@pytest.mark.sampled
def test_t1_projection_consistency(clip):
    orbit_radius = clip.meta["orbit_radius"]
    report, notes = [], []
    for k in range(1, clip.num_objects + 1):
        points, normals = sample_surface(load_canonical_mesh(clip.asset_path(k, "mesh")), 10_000, seed=k)
        for t in range(clip.num_frames):
            m = projection_consistency(
                points, normals, clip.objects[k - 1], clip.K, clip.T_cw[t],
                clip.amodal(k, t), clip.visible(t) == k, clip.depth(t),
            )
            if m["points_in_amodal"] < 0.95:
                report.append(f"k={k} t={t} {m}")
            elif m["amodal_coverage"] < 0.90 or not m["median_depth_err"] < 0.01 * orbit_radius:
                notes.append(f"k={k} t={t} {m}")
    assert not report, "\n".join(report[:10])
    # Coverage and depth error only warn: 10k points under-cover large silhouettes, and the point
    # z-buffer misjudges hollow objects. T1b checks silhouette and depth exactly on the full mesh.
    if notes:
        warnings.warn(f"T1 coverage/depth outside the soft bounds in {len(notes)} object-frames, e.g. {notes[0]}")


def test_t3_visibility_rules(clip, cfg):
    visible = np.stack([clip.visible(t) for t in range(clip.num_frames)])
    amodal = np.stack([[clip.amodal(k, t) for t in range(clip.num_frames)] for k in range(1, clip.num_objects + 1)])
    assert visible.max() <= clip.num_objects
    stats = visibility_stats(visible, amodal)
    failures = visibility_failures(stats, cfg)
    assert not failures, failures[:10]
    np.testing.assert_array_equal(stats["visible_px"], clip.meta["visibility"]["visible_px"])
    # Visible pixels of an object must be part of its amodal silhouette.
    for k in range(clip.num_objects):
        assert not np.any((visible == k + 1) & ~amodal[k])


def test_t4_background_coverage(clip, cfg):
    bg = clip.meta["background"]
    K_bg = np.array(bg["K_bg"])
    ref = np.array(clip.meta["reference_T_cw"])
    ones = np.ones((bg["height"], bg["width"]), np.uint8)
    for t in range(clip.num_frames):
        R_rel = relative_rotation(clip.T_cw[t], ref)
        assert covers(clip.K, R_rel, K_bg, bg["width"], bg["height"], cfg.width, cfg.height, 0.0), f"t={t}"
        _, defined = warp(ones, homography(clip.K, R_rel, K_bg), cfg.width, cfg.height)
        assert defined.all(), f"t={t}: {np.count_nonzero(~defined)} undefined pixels"


def test_format(clip, cfg):
    assert clip.cameras["convention"] == "opencv"
    assert clip.num_frames == cfg.num_frames
    rgb, fg, vis, depth = clip.rgb(0), clip.rgba_fg(0), clip.visible(0), clip.depth(0)
    assert rgb.shape == (cfg.height, cfg.width, 3) and rgb.dtype == np.uint8
    assert fg.shape == (cfg.height, cfg.width, 4) and fg.dtype == np.uint8
    assert vis.shape == (cfg.height, cfg.width) and vis.dtype == np.uint8
    assert depth.dtype == np.float32
    np.testing.assert_array_equal(np.isnan(depth), vis == 0)
    for o in clip.objects:
        assert clip.asset_path(o["k"], "mesh").exists()
        R = o["R_wo"]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.all(o["s_wo"] == o["s_wo"][0])
    for T in clip.T_cw:
        np.testing.assert_allclose(T[3], [0, 0, 0, 1])


@pytest.mark.sampled
def test_t1b_raycast_consistency(clip):
    """Blender's masks and depth must match an independent ray-cast of the GT-posed mesh."""
    report = []
    for k in range(1, clip.num_objects + 1):
        mesh = load_canonical_mesh(clip.asset_path(k, "mesh"))
        for t in range(clip.num_frames):
            m = raycast_consistency(
                mesh, clip.objects[k - 1], clip.K, clip.T_cw[t],
                clip.amodal(k, t), clip.visible(t) == k, clip.depth(t),
            )
            if t1b_fails(m):
                report.append(f"k={k} t={t} {m}")
    assert not report, "\n".join(report[:10])


@pytest.mark.sampled
def test_t1b_detects_convention_errors(clip):
    """Negative controls: a 5 deg rotation, a 0.1*s shift and a half-pixel principal point shift must fail."""
    t = clip.num_frames // 2
    for k in range(1, clip.num_objects + 1):
        mesh = load_canonical_mesh(clip.asset_path(k, "mesh"))
        obj = clip.objects[k - 1]
        masks = (clip.amodal(k, t), clip.visible(t) == k, clip.depth(t))
        check = lambda o, K=clip.K: t1b_fails(raycast_consistency(mesh, o, K, clip.T_cw[t], *masks))
        a = np.radians(5.0)
        tilt = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        shift = 0.1 * obj["s_wo"][0] * clip.T_cw[t][:3, :3].T[:, 0]
        K_shift = clip.K.copy()
        K_shift[0, 2] += 0.5
        assert not check(obj)
        assert check({**obj, "R_wo": obj["R_wo"] @ tilt}), f"k={k}: 5 deg rotation not detected"
        assert check({**obj, "t_wo": obj["t_wo"] + shift}), f"k={k}: 0.1*s shift not detected"
        assert check(obj, K_shift), f"k={k}: half-pixel principal point shift not detected"
