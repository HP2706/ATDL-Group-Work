import blenderproc as bproc  # must be the first line under blenderproc run

# Render one clip inside Blender: `blenderproc run render_clip.py <scene.json> <out.npz>`.
#
# scene.json is written by `sam3dvid.generate`. The output npz holds colors (T,H,W,4 uint8),
# depth (T,H,W float32, z-depth, inf on background), visible (T,H,W uint8 instance map) and
# amodal (N,T,H,W bool).

import json
import sys

import bpy
import numpy as np


def sun_rotation(direction):
    """Rotation whose local +Z points at the sun; Blender suns shine along their local -Z."""
    z = np.asarray(direction, float)
    z /= np.linalg.norm(z)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


def configure_renderer(samples, denoiser):
    bproc.renderer.set_max_amount_of_samples(samples)
    bproc.renderer.set_noise_threshold(0.0)
    bproc.renderer.set_denoiser(denoiser)


def main(scene_path, out_path):
    scene = json.loads(open(scene_path).read())
    bproc.init()
    bpy.context.scene.view_settings.view_transform = "Standard"
    bpy.context.scene.view_settings.look = "None"
    bpy.context.scene.cycles.seed = scene["render_seed"]
    bpy.context.scene.cycles.use_animated_seed = False

    objects = []
    for spec in scene["objects"]:
        loaded = bproc.loader.load_obj(spec["mesh_path"])
        if len(loaded) != 1:
            raise RuntimeError(f"{spec['mesh_path']}: expected one mesh object, got {len(loaded)}")
        obj = loaded[0]
        if obj.blender_obj.parent is not None or not np.allclose(obj.get_local2world_mat(), np.eye(4), atol=1e-6):
            raise RuntimeError(f"{spec['mesh_path']}: imported with a non-identity transform")
        obj.set_local2world_mat(np.array(spec["matrix_world"]))
        obj.set_cp("category_id", spec["k"])
        objects.append(obj)

    light = scene["light"]
    sun = bproc.types.Light(light_type="SUN")
    sun.set_energy(light["sun_strength"])
    sun.set_rotation_mat(sun_rotation(light["sun_direction"]))
    bproc.renderer.set_world_background([1.0, 1.0, 1.0], strength=light["ambient_strength"])

    cam = scene["camera"]
    bproc.camera.set_intrinsics_from_K_matrix(np.array(cam["K"]), cam["width"], cam["height"])
    for M in cam["blender_cam2world"]:
        bproc.camera.add_camera_pose(np.array(M))

    bproc.renderer.set_render_devices(desired_gpu_device_type=["OPTIX", "CUDA"])
    bproc.renderer.set_output_format(enable_transparency=True)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_segmentation_output(map_by=["category_id"], default_values={"category_id": 0})

    configure_renderer(scene["samples"], scene["denoiser"])
    data = bproc.renderer.render()
    colors = np.stack(data["colors"]).astype(np.uint8)
    depth = np.stack(data["depth"]).astype(np.float32)
    visible = np.stack(data["category_id_segmaps"]).astype(np.uint8)

    # Amodal silhouettes: one cheap pass per object with every other object hidden.
    configure_renderer(1, None)
    amodal = []
    for obj, spec in zip(objects, scene["objects"]):
        for other in objects:
            other.hide(other is not obj)
        segs = bproc.renderer.render()["category_id_segmaps"]
        amodal.append(np.stack(segs) == spec["k"])
    for other in objects:
        other.hide(False)

    np.savez_compressed(out_path, colors=colors, depth=depth, visible=visible, amodal=np.stack(amodal))


main(sys.argv[1], sys.argv[2])
