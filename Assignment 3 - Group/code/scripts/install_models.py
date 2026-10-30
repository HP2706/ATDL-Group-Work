"""Install pinned upstream code as libraries, without sibling imports or source vendoring."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
import fire


def install_archive(repository: str, revision: str, name: str,
                    packages: list[str] | None = None, modules: list[str] | None = None) -> None:
    """Package an upstream release that has Python modules but no distribution metadata."""
    with tempfile.TemporaryDirectory(prefix="video3d-library-") as directory:
        root = Path(directory)
        archive = root / "source.tar.gz"
        urllib.request.urlretrieve(f"https://codeload.github.com/{repository}/tar.gz/{revision}", archive)
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                target = (root / member.name).resolve()
                if not target.is_relative_to(root) or not (member.isfile() or member.isdir()):
                    raise ValueError(f"Unsafe archive member: {member.name}")
            source.extractall(root)
        checkout = root / f"{repository.split('/')[-1]}-{revision}"
        packaging = (f"packages = {packages!r}" if packages else f"py-modules = {modules!r}")
        (checkout / "pyproject.toml").write_text(
            '[build-system]\nrequires=["setuptools<81", "wheel"]\nbuild-backend="setuptools.build_meta"\n'
            f'[project]\nname="{name}"\nversion="0.0.1+g{revision[:12]}"\n'
            '[tool.setuptools]\n' + packaging + '\n')
        if packages:
            # V-JEPA uses nested src.* imports; include every subpackage in its wheel.
            (checkout / "pyproject.toml").write_text(
                '[build-system]\nrequires=["hatchling"]\nbuild-backend="hatchling.build"\n'
                f'[project]\nname="{name}"\nversion="0.0.1+g{revision[:12]}"\n'
                f'[tool.hatch.build.targets.wheel]\npackages={packages!r}\n')
        subprocess.run(["uv", "pip", "install", "--python", sys.executable,
                        "--no-deps", "--reinstall", str(checkout)], check=True)


def install(metrics: bool = False, runtime: bool = False) -> None:
    project = Path(__file__).resolve().parents[1]
    pins = tomllib.loads((project / "pyproject.toml").read_text())["tool"]["video3d"]["upstream"]
    subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--no-deps",
                    f"sam3d-objects @ git+https://github.com/facebookresearch/sam-3d-objects.git@{pins['sam3d']}"], check=True)
    import hydra
    if hydra.__version__ != "1.3.2":
        raise RuntimeError("The pinned SAM runtime requires hydra-core==1.3.2")
    # Upstream's documented post-install patch, pinned to the same revision it uses.
    urllib.request.urlretrieve(pins["runtime"]["hydra_patch"], Path(hydra.__file__).parent / "core/utils.py")
    install_archive("facebookresearch/vjepa2", pins["vjepa2"], "atdl-vjepa2", packages=["src", "app"])
    install_archive("google/navi", pins["navi"], "atdl-navi", modules=["data_util"])
    if metrics:
        import torch
        build_environment = dict(os.environ)
        if sys.platform == "linux":
            build_environment.setdefault("CC", "gcc")
            build_environment.setdefault("CXX", "g++")
        if torch.version.cuda is not None and Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin/nvcc").exists():
            build_environment["FORCE_CUDA"] = "1"
        subprocess.run(["uv", "pip", "install", "--python", sys.executable,
                        "--no-build-isolation", "--no-deps",
                        f"pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@{pins['pytorch3d']}"], check=True, env=build_environment)
    if runtime:
        for dependency in [pins["runtime"]["kaolin"], pins["runtime"]["moge"],
                           pins["runtime"]["utils3d"]]:
            subprocess.run(["uv", "pip", "install", "--python", sys.executable,
                            "--no-deps", dependency], check=True)


if __name__ == "__main__":
    fire.Fire(install)
