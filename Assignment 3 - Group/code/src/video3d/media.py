"""Paired reconstruction artifacts shared by both conditioning strategies."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch


def surface_heatmap(visibility_file: Path, directory: Path, frame_ids: list[int] | None = None,
                    prefix: str = "surface_visibility") -> tuple[Path, np.ndarray]:
    """Save three canonical surface views, numeric counts and an interactive colored cloud."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize
    import trimesh

    with np.load(visibility_file) as visibility:
        points = visibility["points"].astype(np.float32)
        observed = visibility["visible"]
    selected = list(range(len(observed))) if frame_ids is None else frame_ids
    if not selected or min(selected) < 0 or max(selected) >= len(observed):
        raise ValueError("Heatmap frame indices must select existing source frames")
    counts = observed[selected].sum(0).astype(np.uint8)
    colors = plt.colormaps["viridis"](np.linspace(0, 1, 256))
    colors[0] = [0.6, 0.6, 0.6, 1.0]
    colormap = ListedColormap(colors)
    normalization = Normalize(vmin=0, vmax=len(selected))
    rgba = colormap(normalization(counts))
    rgb = np.rint(rgba[:, :3] * 255).astype(np.uint8)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / f"{prefix}.npz", points=points, frame_count=counts,
                        seen=counts > 0, frame_ids=np.asarray(selected),
                        geometry="voxel-based surface estimate")
    trimesh.points.PointCloud(points, colors=rgb).export(directory / f"{prefix}.ply")
    style: dict[str, object] = {"font.size":8,"axes.labelsize":9,"axes.titlesize":10,
        "legend.fontsize":7,"xtick.labelsize":7,"ytick.labelsize":7,"figure.titlesize":11,
        "savefig.facecolor":"white","savefig.transparent":False,"figure.facecolor":"white"}
    with plt.rc_context(style):
        figure = plt.figure(figsize=(3.3, 7.0))
        title = "Full clip" if frame_ids is None else "Model input"
        figure.suptitle(f"{title}: surface exposure\n{len(selected)} frames; {(counts > 0).mean():.1%} seen", y=0.99)
        low, high = points.min(0), points.max(0)
        center = (low + high) / 2
        radius = max(float((high - low).max() / 2), 1e-3)
        for index, (label, elevation, azimuth) in enumerate(
                [("Front",20,-90),("Back",20,90),("Top",90,-90)], start=1):
            axes = figure.add_subplot(3, 1, index, projection="3d")
            axes.scatter(*points.T, c=rgba, s=1.2, depthshade=False, linewidths=0)
            axes.view_init(elev=elevation, azim=azimuth)
            axes.set(xlim=(center[0]-radius,center[0]+radius),
                     ylim=(center[1]-radius,center[1]+radius),
                     zlim=(center[2]-radius,center[2]+radius))
            axes.set_box_aspect((1,1,1))
            axes.set_axis_off()
            axes.set_title(label, pad=-8)
        figure.subplots_adjust(top=0.91,bottom=0.16,left=0.02,right=0.98,hspace=0.02)
        colorbar_axes = figure.add_axes((0.2,0.09,0.6,0.016))
        figure.colorbar(plt.cm.ScalarMappable(norm=normalization,cmap=colormap),
                        cax=colorbar_axes,orientation="horizontal",label="Frames in which point is visible")
        figure.text(0.5,0.015,"Gray: never seen. Estimated from 64³ voxels.",ha="center",fontsize=7)
        image = directory / f"{prefix}.png"
        figure.savefig(image,dpi=300)
        plt.close(figure)
    return image, np.column_stack([points, rgb])


def export_mesh(voxels: torch.Tensor, path: Path) -> Path | None:
    import trimesh
    from skimage.measure import marching_cubes
    grid = voxels.detach().cpu().numpy().astype(np.float32)
    if grid.min() == grid.max():
        return None
    vertices, faces, _, _ = marching_cubes(np.pad(grid, 1), level=0.5)
    vertices = (vertices - 1 + 0.5) / np.array(grid.shape) - 0.5
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)
    return path


def save_pair(sample: dict[str, Any], pred: torch.Tensor, truth: torch.Tensor,
              record: dict[str, Any], directory: Path, wandb_run: Any = None,
              visibility_file: Path | None = None,
              frame_ids: list[int] | None = None) -> tuple[Path | None, np.ndarray | None]:
    import imageio.v3 as iio
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=True)
    inputs = sample["inputs"]
    frames = (inputs["frames"].detach().cpu().permute(0,2,3,1).numpy() * 255).clip(0,255).astype(np.uint8)
    iio.imwrite(directory / "input.mp4", frames, fps=8)
    Image.fromarray(inputs["ref_rgb"].cpu().numpy()).save(directory / "reference.png")
    Image.fromarray(inputs["ref_mask"].cpu().numpy().astype(np.uint8)*255).save(directory / "mask.png")
    export_mesh(pred, directory / "prediction.glb")
    export_mesh(truth, directory / "ground_truth.glb")
    (directory / "pairing.json").write_text(json.dumps(record, indent=2, allow_nan=False))
    heatmap_image, heatmap_cloud = (None, None)
    clip_image = None
    if visibility_file is not None:
        heatmap_image, heatmap_cloud = surface_heatmap(visibility_file,directory,frame_ids,"input_visibility")
        clip_image, _ = surface_heatmap(visibility_file,directory,prefix="clip_visibility")
    if wandb_run is not None:
        import wandb
        artifact = wandb.Artifact(directory.name, type="reconstruction", metadata=record)
        artifact.add_dir(str(directory))
        wandb_run.log_artifact(artifact)
        media: dict[str, Any] = {"eval/input":wandb.Video(str(directory / "input.mp4")),
                                 "eval/reference":wandb.Image(str(directory / "reference.png"))}
        for name, grid in [("prediction",pred),("ground_truth",truth)]:
            points = ((grid.nonzero().float().cpu().numpy() + 0.5) / grid.shape[0] - 0.5)
            if len(points):
                media[f"eval/{name}"]=wandb.Object3D(points)
        if heatmap_image is not None:
            media.update({"eval/input_visibility_heatmap":wandb.Image(str(heatmap_image)),
                          "eval/clip_visibility_heatmap":wandb.Image(str(clip_image)),
                          "eval/surface_frame_count":wandb.Object3D(heatmap_cloud)})
        wandb_run.log(media)
    return heatmap_image, heatmap_cloud
