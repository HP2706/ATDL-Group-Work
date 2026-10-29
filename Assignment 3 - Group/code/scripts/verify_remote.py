"""Bounded real-model verification job. Runs the ordinary portable verification script."""
from __future__ import annotations
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import modal

CODE = Path(os.environ.get("VIDEO3D_SOURCE_DIR", str(Path(__file__).resolve().parents[1])))
run_name = "unified-verification-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
app = modal.App(run_name)
cache = modal.Volume.from_name("atdl-unified-verification", create_if_missing=True)
if modal.is_local():
    image = (
        modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
        .apt_install("curl", "xz-utils", "git", "build-essential", "libegl1", "libgl1", "libusb-1.0-0", "libglib2.0-0",
                     "libxrender1", "libsm6", "libxext6", "libxi6", "libxxf86vm1",
                     "libxfixes3", "libxkbcommon0", "ninja-build", "ffmpeg")
        .uv_pip_install("uv", "setuptools<81", "wheel", "ninja", "packaging<25")
        .run_commands("mkdir -p /opt/blender && curl -fL https://download.blender.org/release/Blender4.5/blender-4.5.15-linux-x64.tar.xz -o /opt/blender/source.tar.xz && tar -xJf /opt/blender/source.tar.xz -C /opt/blender")
        .env({"MAX_JOBS":"4", "TORCH_CUDA_ARCH_LIST":"10.0", "BUILD_NO_CUDA":"1", "CUDA_HOME":"/usr/local/cuda"})
        .uv_sync(str(CODE), groups=["gpu_runtime", "generation", "dev"])
        .add_local_dir(CODE, "/opt/video3d", copy=True,
                       ignore=[".venv", ".env", ".env.*", "__pycache__", "*.pyc", "notebooks", "historical-tests", "outputs", "scratch", "cache"])
        .env({"MAX_JOBS":"4", "TORCH_CUDA_ARCH_LIST":"10.0", "BUILD_NO_CUDA":"1",
              "LIDRA_SKIP_INIT":"1", "ATTN_BACKEND":"sdpa", "CC":"gcc", "CXX":"g++"})
        .run_commands("cd /opt/video3d && uv pip install --python $(which python) --no-deps . && python scripts/install_models.py --metrics=True --runtime=True")
    )


    image = image.uv_pip_install(*['dataclasses-json==0.6.7', 'deprecated==1.3.1', 'einops-exts==0.0.4', 'fasteners==0.20', 'ftfy==6.3.1', 'fvcore==0.1.5.post20221221', 'ipycanvas==0.14.3', 'ipyevents==2.0.4', 'marshmallow==3.26.2', 'mypy-extensions==1.1.0', 'openexr==3.5.2', 'point-cloud-utils==0.34.0', 'pycocotools==2.0.11', 'pygltflib==1.16.5', 'python-dotenv==1.2.4', 'randomname==0.2.1', 'roma==1.8.0', 'rootutils==1.0.7', 'rtree==1.4.1', 'tabulate==0.10.0', 'typing-inspect==0.9.0', 'usd-core==26.11', 'warp-lang==1.18.0', 'wrapt==2.5.1', 'yacs==0.1.8'])

    image = image.add_local_file(Path(__file__).parent / "install_models.py", "/opt/install_models_refresh.py", copy=True)
    image = image.run_commands("python -c \"import runpy; helpers=runpy.run_path('/opt/install_models_refresh.py'); helpers['install_archive']('facebookresearch/vjepa2', '204698b45b3712590f06245fbfba32d3be539812', 'atdl-vjepa2', packages=['src','app'])\"")

    # Overlay the final source after the costly dependency image, so fixes reuse compiled libraries.
    VERIFY_SOURCE = Path(os.environ.get("VIDEO3D_TEST_SOURCE_DIR", str(CODE)))
    image = image.add_local_dir(VERIFY_SOURCE, "/workspace", copy=False,
        ignore=[".venv", ".env", ".env.*", "__pycache__", "*.pyc", "notebooks", "historical-tests", "outputs", "scratch", "cache"])
else:
    image = modal.Image.debian_slim()


def install_project() -> None:
    subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--no-deps", "/workspace"], check=True)


@app.function(image=image, cpu=4, memory=8192, timeout=900, scaledown_window=2,
              volumes={"/cache":cache})
def verify_cpu(run_id: str, dataset: str = "") -> dict[str, object]:
    install_project()
    directory = Path("/cache") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    try:
        selection = ["tests/generation", "-m", "not slow"] if dataset else []
        environment = {**os.environ,"OMP_NUM_THREADS":"2"}
        if dataset:
            environment["SAM3DVID_DATASET"] = dataset
        result = subprocess.run([sys.executable, "-m", "pytest", *selection, "-q", "--tb=short",
                                  "--junitxml", str(directory / "cpu-tests.xml")],
                                 cwd="/workspace", env=environment,
                                 capture_output=True, text=True, timeout=800)
        (directory / "cpu-tests.log").write_text(result.stdout + result.stderr)
        print(result.stdout[-14000:] + result.stderr[-2000:])
        if result.returncode:
            raise RuntimeError(f"CPU verification failed: {directory}")
        return {"returncode":0,"report":str(directory / "cpu-tests.xml")}
    finally:
        cache.commit()


@app.function(image=image, gpu="L40S", cpu=8, memory=49152, timeout=1800,
              scaledown_window=2, max_containers=1,
              volumes={"/cache":cache},
              secrets=[modal.Secret.from_dotenv(os.environ["VIDEO3D_VERIFICATION_ENV"])] if modal.is_local() else [modal.Secret.from_dict({})])
def verify(run_id: str) -> dict[str, object]:
    install_project()
    directory = Path("/cache") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "HF_HOME":"/cache/huggingface", "TORCH_HOME":"/cache/torch",
                   "OMP_NUM_THREADS":"4", "OPENBLAS_NUM_THREADS":"4"}
    try:
        with (directory / "verification.log").open("w") as log:
            result = subprocess.run([sys.executable, "-m", "scripts.verify_models", "--output", str(directory)],
                                    cwd="/workspace", env=environment, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=1500)
        print((directory / "verification.log").read_text()[-16000:])
        summary: dict[str, object] = {"returncode":result.returncode,"output":str(directory)}
        (directory / "job.json").write_text(json.dumps(summary))
        if result.returncode:
            raise RuntimeError(f"Real-model verification failed: {directory}")
        return summary
    finally:
        cache.commit()


@app.function(image=image, cpu=1, memory=1024, timeout=60)
def dependencies() -> None:
    import importlib.metadata as metadata
    installed = {d.metadata["Name"].lower().replace("_", "-") for d in metadata.distributions()}
    from packaging.requirements import Requirement
    for name in ["kaolin", "moge", "sam3d-objects", "pytorch3d"]:
        print(name, metadata.requires(name))
        for value in metadata.requires(name) or []:
            requirement = Requirement(value)
            if requirement.marker is None or requirement.marker.evaluate():
                if requirement.name.lower().replace("_", "-") not in installed:
                    print("MISSING", str(requirement))


@app.function(image=image, cpu=2, memory=4096, timeout=180)
def import_runtime() -> None:
    from importlib import import_module
    import_module("sam3d_objects.pipeline.inference_pipeline_pointmap")
    print("SAM runtime imports successfully", flush=True)


@app.function(image=image, gpu="L40S", cpu=4, memory=16384, timeout=600,
              scaledown_window=2, volumes={"/cache":cache})
def render_check(directory: str) -> None:
    install_project()
    try:
        result = subprocess.run([sys.executable,"-m","scripts.verify_generation","--output",directory],
                                cwd="/workspace", timeout=540,
                                env={**os.environ,"OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2"})
        if result.returncode:
            raise RuntimeError("Generator verification failed")
    finally:
        cache.commit()


@app.local_entrypoint()
def main(cpu_only: bool = False, gpu_only: bool = False, deps_only: bool = False, imports_only: bool = False, dataset: str = "", render: str = "") -> None:
    with modal.enable_output():
        if render:
            render_check.remote(render)
            return
        if imports_only:
            import_runtime.remote()
            return
        if deps_only:
            dependencies.remote()
            return
        if not gpu_only:
            print(verify_cpu.remote(run_name, dataset), flush=True)
        if not cpu_only:
            print(verify.remote(run_name), flush=True)
