import blenderproc as bproc  # must be the first line under blenderproc run

# Thumbnails of normalized assets: `blenderproc run render_thumbs.py <jobs.json>`.
#
# jobs.json is a list of {"mesh": <mesh.glb>, "out": <thumb.npy>}. Each asset is shown from a fixed
# three-quarter view above the horizon with +Z up, so a thumbnail also shows the canonical
# orientation. Output is RGBA uint8 (SIZE, SIZE, 4) with a transparent background.

import json
import sys

import bpy
import numpy as np

SIZE = 256
HFOV_DEG = 35.0
DISTANCE = 2.7
AZIMUTH_DEG = -50.0
ELEVATION_DEG = 25.0
SAMPLES = 24


def render(job: dict) -> None:
    bproc.clean_up(clean_up_camera=True)
    bproc.loader.load_obj(job["mesh"])
    sun = bproc.types.Light(light_type="SUN")
    sun.set_energy(3.0)
    sun.set_rotation_mat(bproc.camera.rotation_from_forward_vec(np.array([-0.4, 0.5, -1.0])))
    bproc.renderer.set_world_background([1.0, 1.0, 1.0], strength=0.6)

    f = (SIZE / 2) / np.tan(np.radians(HFOV_DEG) / 2)
    K = np.array([[f, 0, (SIZE - 1) / 2], [0, f, (SIZE - 1) / 2], [0, 0, 1.0]])
    bproc.camera.set_intrinsics_from_K_matrix(K, SIZE, SIZE)
    az, el = np.radians(AZIMUTH_DEG), np.radians(ELEVATION_DEG)
    eye = DISTANCE * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    R = bproc.camera.rotation_from_forward_vec(-eye)
    bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(eye, R))
    data = bproc.renderer.render()
    np.save(job["out"], np.asarray(data["colors"][0]).clip(0, 255).astype(np.uint8))


def main(jobs_path: str) -> None:
    bproc.init()
    bpy.context.scene.view_settings.view_transform = "Standard"
    bproc.renderer.set_max_amount_of_samples(SAMPLES)
    bproc.renderer.set_noise_threshold(0.0)
    bproc.renderer.set_output_format(enable_transparency=True)
    try:
        bproc.renderer.set_render_devices(desired_gpu_device_type=["OPTIX", "CUDA"])
        # Keep the CPU free for parallel jobs when a GPU is in use.
        for device in bpy.context.preferences.addons["cycles"].preferences.devices:
            device.use = device.type != "CPU"
    except Exception:
        pass  # no GPU: Cycles renders on the CPU
    with open(jobs_path) as f:
        jobs = json.load(f)
    for job in jobs:
        try:
            render(job)
            print(f"[thumbs] ok {job['out']}", flush=True)
        except Exception as e:
            print(f"[thumbs] FAILED {job['out']}: {e!r}", flush=True)


main(sys.argv[1])
