from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from video3d.media import surface_heatmap


def test_heatmap_uses_selected_frames_and_preserves_unseen_points(tmp_path: Path) -> None:
    points = np.array([[-.5,-.5,-.5],[.5,-.5,.5],[-.5,.5,.5],[.5,.5,-.5]],dtype=np.float32)
    visible = np.array([[True,False,False,False],[True,True,True,False],
                        [False,True,False,False]],dtype=bool)
    source = tmp_path / "source.npz"
    np.savez(source,points=points,visible=visible)
    image, cloud = surface_heatmap(source,tmp_path,[0,2],"input")
    with np.load(tmp_path / "input.npz") as output:
        assert output["frame_count"].tolist() == [1,1,0,0]
        assert output["seen"].tolist() == [True,True,False,False]
    assert cloud.shape == (4,6)
    np.testing.assert_array_equal(cloud[2,3:],[153,153,153])
    assert (tmp_path / "input.ply").stat().st_size > 0
    with Image.open(image) as rendered:
        assert rendered.size == (990,2100)
    surface_heatmap(source,tmp_path,prefix="full")
    with np.load(tmp_path / "full.npz") as output:
        assert output["frame_count"].tolist() == [2,2,1,0]
