import blenderproc as bproc  # must be the first line under blenderproc run

# Normalize assets inside Blender: `blenderproc run normalize_assets.py <jobs.json>`.
#
# jobs.json is a list of {"source": <mesh file>, "out_dir": <dir>}. For each job this writes
# mesh.glb (single object, identity transform, canonical frame), mesh.ply (TRELLIS voxelizer
# input) and normalize.json (scale, offset, original bbox). `normalize_scene` and `scene_bbox`
# are copied verbatim from TRELLIS dataset_toolkits/blender_script/render.py.

import json
import math
import os
import sys
from typing import Tuple

import bpy
from mathutils import Vector


def init_scene() -> None:
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.textures, bpy.data.images):
        for block in list(collection):
            collection.remove(block, do_unlink=True)


def load_object(path: str) -> None:
    ext = path.rsplit(".", 1)[-1].lower()
    if ext in {"glb", "gltf"}:
        bpy.ops.import_scene.gltf(filepath=path, merge_vertices=True, import_shading="NORMALS")
    elif ext == "obj":
        # GSO OBJs are Z-up already (Gazebo convention), so import without an axis change.
        # TRELLIS never imports GSO; its Y-up OBJ default would lay these objects on their side.
        bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
    else:
        raise ValueError(f"unsupported mesh format: {path}")


def scene_bbox() -> Tuple[Vector, Vector]:
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene() -> Tuple[float, Vector]:
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(scene_root_objects) > 1:
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale

    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

    return scale, offset


def bake_to_single_mesh() -> None:
    """Convert everything to one triangulated mesh object with its transform applied.

    Triangulation matches TRELLIS's mesh.ply export (BEAUTY quads and ngons).
    """
    for obj in bpy.context.scene.objects:
        obj.hide_set(False)
    bpy.ops.object.select_all(action="SELECT")
    bpy.context.view_layer.objects.active = next(o for o in bpy.context.scene.objects if o.type == "MESH")
    bpy.ops.object.convert(target="MESH")
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    for obj in [o for o in bpy.context.scene.objects if o.type != "MESH"]:
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.join()
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.reveal()
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")


def process(job: dict) -> None:
    init_scene()
    load_object(job["source"])
    raw_min, raw_max = scene_bbox()
    scale, offset = normalize_scene()
    bake_to_single_mesh()
    norm_min, norm_max = scene_bbox()

    out = job["out_dir"]
    os.makedirs(out, exist_ok=True)
    bpy.ops.wm.ply_export(
        filepath=os.path.join(out, "mesh.ply"),
        forward_axis="Y",
        up_axis="Z",
        apply_modifiers=True,
        export_triangulated_mesh=True,
        export_normals=False,
        export_uv=False,
        export_colors="NONE",
        ascii_format=False,
    )
    bpy.ops.export_scene.gltf(filepath=os.path.join(out, "mesh.glb"), export_format="GLB", export_apply=True)
    mesh = bpy.context.scene.objects[0].data
    info = {
        "scale": scale,
        "offset": list(offset),
        "raw_bbox_min": list(raw_min),
        "raw_bbox_max": list(raw_max),
        "normalized_bbox_min": list(norm_min),
        "normalized_bbox_max": list(norm_max),
        "num_vertices": len(mesh.vertices),
        "num_faces": len(mesh.polygons),
        "num_materials": len(mesh.materials),
    }
    with open(os.path.join(out, "normalize.json"), "w") as f:
        json.dump(info, f, indent=1)


def main(jobs_path: str) -> None:
    bproc.init()
    with open(jobs_path) as f:
        jobs = json.load(f)
    for job in jobs:
        try:
            process(job)
            print(f"[normalize] ok {job['out_dir']}", flush=True)
        except Exception as e:
            print(f"[normalize] FAILED {job['out_dir']}: {e!r}", flush=True)


main(sys.argv[1])
