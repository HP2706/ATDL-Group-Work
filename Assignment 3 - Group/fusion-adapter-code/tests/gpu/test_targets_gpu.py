"""T5 (VAE round trip) and T4 (canonical frame test) on real eval assets."""

import numpy as np
import pytest
import torch

from vidsam3d.data import reader
from vidsam3d.data.targets import fill_voxels
from vidsam3d.inspect_sam3d import frame_test
from vidsam3d.metrics import voxel_iou

pytestmark = pytest.mark.gpu


def test_t5_vae_round_trip(gpu_cfg, sam3d):
    voxels_file = reader.open_split(f"{gpu_cfg.paths.data_root}/{gpu_cfg.data.eval_split}").voxels_file
    assets = sorted(reader.voxel_table(str(voxels_file)))[:50]
    results = {}
    for fill in ("surface", "solid"):
        ious = []
        for asset_id in assets:
            vox = reader.load_voxels(voxels_file, asset_id)
            vox = fill_voxels(vox) if fill == "solid" else vox
            v = torch.from_numpy(vox)[None].cuda()
            ious.append(voxel_iou(sam3d.decode_latent(sam3d.encode_voxels(v))[0], v[0]))
        results[fill] = float(np.mean(ious))
    print(f"T5 mean round-trip IoU on {len(assets)} eval assets: {results}")
    assert max(results.values()) >= 0.9, results


def test_t4_frame_test_runs(gpu_cfg, sam3d):
    report = frame_test(gpu_cfg, num_objects=5, model=sam3d)
    print(f"T4 (5 objects): winner {report['winner']} share {report['winner_share']:.2f}")
    assert len(report["wins_per_rotation"]) == 24
