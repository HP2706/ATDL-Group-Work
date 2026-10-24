from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any
import modal

app = modal.App("mesh-comparison-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), tags={"project": "atdl-video-to-3d", "experiment": "notebook-mesh-comparison"})
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "trimesh", "fast-simplification", "matplotlib", "pillow")


@app.function(image=image, cpu=2, memory=4096, timeout=600, scaledown_window=2)
def render(mesh_files: dict[str, bytes], objects: list[dict[str, Any]]) -> dict[str, bytes]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np
    from PIL import Image
    import trimesh

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.facecolor": "white", "savefig.facecolor": "white"})
    outputs: dict[str, bytes] = {}
    counts: list[dict[str, Any]] = []
    for obj in objects:
        identity = obj["example_id"]
        models = [{"key": "ground_truth", "label": "Ground truth", "mesh": obj["ground_truth.glb"], "metrics": {}}] + obj["models"]
        rows: list[Image.Image] = []
        for model in models:
            mesh = trimesh.load(io.BytesIO(mesh_files[model["mesh"]]), file_type="glb", force="mesh")
            original_faces = len(mesh.faces)
            if original_faces > 20000:
                mesh = mesh.simplify_quadric_decimation(face_count=20000)
            outputs[f"preview/{identity}/{model['key']}.glb"] = mesh.export(file_type="glb")
            triangles = mesh.vertices[mesh.faces]
            light = np.array([0.3, -0.4, 0.85]); light /= np.linalg.norm(light)
            intensity = 0.4 + 0.45 * np.clip(mesh.face_normals @ light, 0, 1)
            colors = np.repeat(intensity[:, None], 3, axis=1)
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), subplot_kw={"projection": "3d"})
            for ax, azimuth, label in zip(axes, [-60, 120], ["View 1", "Opposite view"], strict=True):
                ax.add_collection3d(Poly3DCollection(triangles, facecolors=colors, edgecolors="none", linewidths=0, rasterized=True))
                ax.set(xlim=(-0.58, 0.58), ylim=(-0.58, 0.58), zlim=(-0.58, 0.58))
                ax.set_box_aspect((1, 1, 1)); ax.set_proj_type("ortho")
                ax.view_init(elev=22, azim=azimuth); ax.set_axis_off(); ax.set_title(label, color="#777777", fontsize=8)
            metric = model["metrics"].get("chamfer")
            title = model["label"] + (f"     Chamfer {metric:.4f}" if metric is not None else "")
            fig.suptitle(title, x=0.03, ha="left", y=0.98, fontsize=11, fontweight="semibold", color="#292929")
            fig.subplots_adjust(left=0, right=1, bottom=0, top=0.87, wspace=0)
            buffer = io.BytesIO(); fig.savefig(buffer, format="png", dpi=150); plt.close(fig)
            outputs[f"preview/{identity}/{model['key']}.png"] = buffer.getvalue()
            rows.append(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"))
            counts.append({"example_id": identity, "regime": model["key"], "source_faces": original_faces, "display_faces": len(mesh.faces)})
        contact = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "white")
        offset = 0
        for row in rows:
            contact.paste(row, (0, offset)); offset += row.height
        buffer = io.BytesIO(); contact.save(buffer, format="PNG", optimize=True)
        outputs[f"preview/{identity}/contact.png"] = buffer.getvalue()
        print("Rendered same-camera comparison:", identity, flush=True)
    outputs["preview/manifest.json"] = json.dumps(counts, indent=2).encode()
    return outputs


def main(directory: str = "results/notebook-comparison") -> None:
    root = Path(directory)
    bundle = json.loads((root / "comparison.json").read_text())
    paths = {obj["ground_truth.glb"] for obj in bundle["objects"]}
    paths.update(model["mesh"] for obj in bundle["objects"] for model in obj["models"])
    mesh_files = {path: (root / path).read_bytes() for path in paths}
    with modal.enable_output(), app.run():
        outputs = render.remote(mesh_files, bundle["objects"])
        for name, content in outputs.items():
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(content)
    print(f"Saved {len(outputs)} rendered assets; remote CPU allocation released.")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
