"""Bounded Linux/GPU checks using the portable project commands."""
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


    # Overlay the final source after the costly dependency image, so fixes reuse compiled libraries.
    VERIFY_SOURCE = Path(os.environ.get("VIDEO3D_TEST_SOURCE_DIR", str(CODE)))
    image = image.add_local_dir(VERIFY_SOURCE, "/workspace", copy=False,
        ignore=[".venv", ".env", ".env.*", "__pycache__", "*.pyc", "notebooks", "historical-tests", "outputs", "scratch", "cache"])
else:
    image = modal.Image.debian_slim()


def install_project() -> None:
    subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--no-deps", "/workspace"], check=True)


def project_tests(run_id: str, dataset: str = "", gpu_data: str = "", slow: bool = False) -> dict[str, object]:
    install_project()
    directory = Path("/cache") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    try:
        selection = ["tests/generation/test_t5_determinism.py"] if slow else (["tests/gpu"] if gpu_data else (["tests/generation", "-m", "not slow"] if dataset else []))
        environment = {**os.environ,"OMP_NUM_THREADS":"2"}
        if dataset:
            environment["SAM3DVID_DATASET"] = dataset
        if gpu_data:
            environment.update(DATA_ROOT=gpu_data, SAM3D_CHECKPOINT_DIR="/cache/huggingface/hub/models--facebook--sam-3d-objects/snapshots/2e73555018d2741ccd486e56c24fac41155a1dc6/checkpoints",
                HF_HOME="/cache/huggingface", TORCH_HOME="/cache/torch",
                OUTPUT_ROOT=str(directory/"outputs"), SCRATCH_ROOT=str(directory/"scratch"), CACHE_ROOT=str(directory/"cache"))
        result = subprocess.run([sys.executable, "-m", "pytest", *selection, "-q", "--tb=short",
                                  "--junitxml", str(directory / "cpu-tests.xml")],
                                 cwd="/workspace", env=environment,
                                 capture_output=True, text=True, timeout=800)
        (directory / "cpu-tests.log").write_text(result.stdout + result.stderr)
        print(result.stdout[-14000:] + result.stderr[-2000:])
        if result.returncode:
            raise RuntimeError(f"Verification failed: {directory}")
        return {"returncode":0,"report":str(directory / "cpu-tests.xml")}
    finally:
        cache.commit()


@app.function(image=image, cpu=4, memory=8192, timeout=900, scaledown_window=2, volumes={"/cache":cache})
def verify_cpu(run_id: str, dataset: str = "") -> dict[str, object]:
    return project_tests(run_id, dataset=dataset)


@app.function(image=image, gpu="L40S", cpu=4, memory=32768, timeout=900,
              scaledown_window=2, volumes={"/cache":cache})
def gpu_tests(run_id: str, gpu_data: str = "", dataset: str = "", slow: bool = False) -> dict[str, object]:
    return project_tests(run_id, gpu_data=gpu_data, dataset=dataset, slow=slow)


@app.function(image=image, gpu="L40S", cpu=8, memory=49152, timeout=1800,
              scaledown_window=2, max_containers=1,
              volumes={"/cache":cache},
              secrets=[modal.Secret.from_dotenv(os.environ["VIDEO3D_VERIFICATION_ENV"])] if modal.is_local() and os.environ.get("VIDEO3D_VERIFICATION_ENV") else [modal.Secret.from_dict({})])
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
def main(cpu_only: bool = False, gpu_only: bool = False, dataset: str = "", render: str = "", gpu_data: str = "", determinism_dataset: str = "") -> None:
    with modal.enable_output():
        if determinism_dataset:
            print(gpu_tests.remote(run_name, dataset=determinism_dataset, slow=True), flush=True)
            return
        if gpu_data:
            print(gpu_tests.remote(run_name, gpu_data), flush=True)
            return
        if render:
            render_check.remote(render)
            return
        if not gpu_only:
            print(verify_cpu.remote(run_name, dataset), flush=True)
        if not cpu_only:
            print(verify.remote(run_name), flush=True)
