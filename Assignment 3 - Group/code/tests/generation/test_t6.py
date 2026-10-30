"""T6 input export: the GT pointmap matches K and SAM 3D's PyTorch3D camera axes."""
import typing

import numpy as np

from video3d.data.generation.conventions import intrinsics, project
from video3d.data.generation.t6 import PYTORCH3D_FROM_OPENCV, pointmap_opencv


def test_pointmap_reprojects_to_its_pixel() -> typing.Any:
    K = intrinsics(512, 512, 50.0)
    depth = np.random.default_rng(0).uniform(1.0, 5.0, (512, 512))
    depth[:10] = np.nan
    points = pointmap_opencv(depth, K)
    finite = np.isfinite(depth)
    uv, z = project(K, points[finite])
    v, u = np.nonzero(finite)
    np.testing.assert_allclose(uv, np.stack([u, v], axis=1), atol=1e-9)
    np.testing.assert_allclose(z, depth[finite])
    assert np.isnan(points[~finite]).all()


def test_pytorch3d_axes() -> typing.Any:
    # PyTorch3D cameras have x left and y up; OpenCV has x right and y down; both look along +z.
    p =np.array([1.0, 2.0, 3.0]) @ PYTORCH3D_FROM_OPENCV.T
    np.testing.assert_array_equal(p, [-1.0, -2.0, 3.0])
