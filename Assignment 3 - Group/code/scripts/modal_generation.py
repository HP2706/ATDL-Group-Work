"""Thin Modal map launcher for the portable synthetic generator."""
from __future__ import annotations

from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time
from typing import Any

import modal

CODE = Path(__file__).resolve().parents[1]
RUN_NAME = "occlusion-pilot-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
app = modal.App(RUN_NAME)
artifacts = modal.Volume.from_name("atdl-occlusion-generation", create_if_missing=True)
sources = modal.Volume.from_name("video-object-dataset")


base_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("curl", "xz-utils", "git", "libegl1", "libgl1", "libusb-1.0-0", "libglib2.0-0",
                 "libxrender1", "libsm6", "libxext6", "libxi6", "libxxf86vm1", "libxfixes3", "libxkbcommon0", "ffmpeg")
    .uv_pip_install("numpy==1.26.4", "blenderproc==2.8.0", "open3d==0.18.0", "opencv-python-headless==4.11.0.86",
                    "trimesh>=4.5,<5", "embreex", "rtree", "chz", "fire", "pydantic", "scipy", "pillow", "imageio[ffmpeg]")
    .run_commands("mkdir -p /opt/blender && curl -fL https://download.blender.org/release/Blender4.5/blender-4.5.15-linux-x64.tar.xz -o /tmp/blender.tar.xz && tar -xJf /tmp/blender.tar.xz -C /opt/blender")
    .add_local_file(CODE / "src/video3d/data/generation/blender/warmup.py", "/opt/blenderproc-warmup.py", copy=True)
    .run_commands("blenderproc run --custom-blender-path /opt/blender/blender-4.5.15-linux-x64 /opt/blenderproc-warmup.py")
    .env({"PYTHONPATH": "/workspace/src", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
          "CUDA_CACHE_DISABLE": "0", "CUDA_CACHE_MAXSIZE": "2147483648", "CUDA_CACHE_PATH": "/opt/cuda-cache"})
    .uv_pip_install("zstandard", "nvidia-cuda-nvcc-cu12")
    .add_local_file(CODE / "scripts/compile_a100_kernel.py", "/opt/compile_a100_kernel.py", copy=True)
    .run_commands("python /opt/compile_a100_kernel.py")
)
image = (base_image.add_local_dir(CODE / "src", "/workspace/src", ignore=["__pycache__", "*.pyc"])
    .add_local_dir(CODE / "scripts", "/workspace/scripts", ignore=["__pycache__", "*.pyc"]))


@app.function(image=image, cpu=1, memory=1024, timeout=60, scaledown_window=2)
def check_image() -> str:
    return subprocess.run(["/opt/blender/blender-4.5.15-linux-x64/blender", "--version"],
                          check=True, capture_output=True, text=True).stdout.splitlines()[0]


@app.function(image=image, gpu="A100", cpu=4, memory=8192, timeout=120, scaledown_window=2)
def check_cuda() -> str:
    import ctypes
    driver = ctypes.CDLL("libcuda.so.1")
    print("CUDA_DRIVER_INIT", driver.cuInit(0), flush=True)
    code = '''import bpy,time
print("BLENDER_STARTED",flush=True)
bpy.ops.wm.read_factory_settings(use_empty=False)
s=bpy.context.scene
s.render.engine="CYCLES"
p=bpy.context.preferences.addons["cycles"].preferences
p.compute_device_type="CUDA"
print("GET_DEVICES",flush=True)
p.get_devices()
print("DEVICES",[(d.name,d.type) for d in p.devices],flush=True)
for device in p.devices:
    device.use=device.type!="CPU"
s.cycles.device="GPU"
s.cycles.samples=1
s.cycles.use_denoising=False
s.render.resolution_x=s.render.resolution_y=32
s.render.resolution_percentage=100
start=time.time()
print("RENDER_START",flush=True)
bpy.ops.render.render()
print("CUDA_SMOKE_SECONDS",time.time()-start)
'''
    result = subprocess.run(["timeout", "--kill-after=5", "45", "/opt/blender/blender-4.5.15-linux-x64/blender", "-b", "-t", "4", "--python-expr", code],
                            capture_output=True, text=True, timeout=60)
    print(result.stdout[-4000:] + result.stderr[-1000:], flush=True)
    result.check_returncode()
    return result.stdout[-200:]


@app.function(image=image, cpu=4, memory=16384, timeout=1200, scaledown_window=2,
              max_containers=1, volumes={"/artifacts": artifacts, "/sources": sources})
def prepare(run_name: str, specification: str = "{}") -> dict[str, Any]:
    root = Path("/artifacts") / run_name
    root.mkdir(parents=True, exist_ok=True)
    (root / "run-specification.json").write_text(specification)
    with (root / "prepare.log").open("w") as log:
        result = subprocess.run(["python", "/workspace/scripts/occlusion_pilot.py", "prepare", str(root)],
                                stdout=log, stderr=subprocess.STDOUT, timeout=1100)
    artifacts.commit()
    print((root / "prepare.log").read_text()[-8000:], flush=True)
    if result.returncode:
        raise RuntimeError(f"Asset preparation failed: {root}")
    return json.loads((root / "preparation.json").read_text())


@app.function(image=image, gpu="A100", cpu=4, memory=16384, timeout=900,
              max_containers=10, scaledown_window=2, volumes={"/artifacts": artifacts})
def generate(seed: int, run_name: str) -> dict[str, Any]:
    artifacts.reload()
    root = Path("/artifacts") / run_name
    log_path = root / f"worker-{seed}.log"
    started = time.time()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.strip()
    print(f"START seed={seed} gpu={gpu}", flush=True)
    with log_path.open("w") as log:
        result = subprocess.run(["python", "/workspace/scripts/occlusion_pilot.py", "generate", str(root), str(seed)],
                                stdout=log, stderr=subprocess.STDOUT, timeout=840)
    status = {"seed": seed, "gpu": gpu, "started": started, "finished": time.time(), "returncode": result.returncode}
    (root / f"worker-{seed}.json").write_text(json.dumps(status))
    artifacts.commit()
    if result.returncode:
        print(log_path.read_text()[-10000:], flush=True)
        status["error"] = log_path.read_text()[-4000:]
        return status
    status["metrics"] = json.loads((root / "previews" / f"{seed:08d}.json").read_text())
    print(f"FINISH seed={seed} seconds={status['finished']-started:.1f}", flush=True)
    return status


@app.function(image=image, cpu=2, memory=4096, timeout=300, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def collect(run_name: str) -> bytes:
    artifacts.reload()
    root = Path("/artifacts") / run_name
    result = subprocess.run(["python", "/workspace/scripts/occlusion_pilot.py", "finalize", str(root)],
                            capture_output=True, text=True, timeout=240)
    print(result.stdout[-4000:] + result.stderr[-1000:], flush=True)
    artifacts.commit()
    result.check_returncode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:") as archive:
        for path in sorted((root / "previews").glob("*")):
            archive.add(path, arcname=f"previews/{path.name}")
        for name in ("config.json", "preparation.json", "summary.json", "run-specification.json", "manifest.jsonl", "validation.json", "export/train/index.json"):
            archive.add(root / name, arcname=name)
    return buffer.getvalue()


@app.local_entrypoint()
def main(run_name: str = "", prepare_only: bool = False, skip_prepare: bool = False,
         image_only: bool = False, cuda_only: bool = False, collect_only: bool = False) -> None:
    run_name = run_name or RUN_NAME
    start = time.time()
    digest = hashlib.sha256()
    for path in sorted([*CODE.joinpath("src").rglob("*.py"), *CODE.joinpath("scripts").rglob("*.py")]):
        digest.update(str(path.relative_to(CODE)).encode())
        digest.update(path.read_bytes())
    specification = {"run_name": run_name, "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE,
        check=True, capture_output=True, text=True).stdout.strip(), "source_sha256": digest.hexdigest(),
        "clips": 10, "frames_per_clip": 32, "resolution": [512, 512], "samples_per_pixel": 16,
        "gpu": "A100 (40GB or automatic 80GB upgrade)", "max_parallel_workers": 10, "worker_timeout_seconds": 900,
        "seeds": list(range(2000000, 2000010)), "model_training": False,
        "acceptance": "30-70% reference occlusion; >=8 frames at >=20%; >=75% visibility somewhere; >=30% recovery of reference-hidden mesh samples"}
    with modal.enable_output():
        if collect_only:
            packed = collect.remote(run_name)
            output = CODE / "results" / run_name
            output.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(packed), mode="r:") as archive:
                archive.extractall(output, filter="data")
            print(f"Saved verified dataset previews: {output}", flush=True)
            return
        if cuda_only:
            print(check_cuda.remote(), flush=True)
            return
        if image_only:
            print(check_image.remote(), flush=True)
            return
        if not skip_prepare:
            print(prepare.remote(run_name, json.dumps(specification, indent=2)), flush=True)
        if prepare_only:
            return
        mapped = list(generate.map(range(2000000, 2000010), [run_name] * 10, order_outputs=False, return_exceptions=True))
        results = [item if isinstance(item, dict) else {"returncode": -1, "error": str(item)} for item in mapped]
        summary = {"run_name": run_name, "workers": results, "elapsed_seconds": time.time() - start,
                   "successful_clips": sum(row["returncode"] == 0 for row in results)}
        # Persist the small summary before retrieving preview artifacts.
        with artifacts.batch_upload() as batch:
            payload = io.BytesIO(json.dumps(summary, indent=2).encode())
            batch.put_file(payload, f"/{run_name}/summary.json")
        packed = collect.remote(run_name)
    output = CODE / "results" / run_name
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(packed), mode="r:") as archive:
        archive.extractall(output, filter="data")
    print(f"Saved previews: {output}", flush=True)
    if summary["successful_clips"] != 10:
        raise RuntimeError(f"Only {summary['successful_clips']}/10 clips succeeded; diagnostics preserved in {output}")
