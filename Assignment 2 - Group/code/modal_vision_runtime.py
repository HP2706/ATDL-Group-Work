"""Shared Modal image and persistent storage for vision experiments."""

from pathlib import Path

import modal


CODE_DIR = Path(__file__).resolve().parent
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.0",
        "torchvision==0.29.0",
        "numpy==2.5.3",
        "pydantic==2.13.5",
        "pandas==3.0.6",
        "pyarrow==25.0.1",
        "fire==0.7.1",
        "tqdm==4.70.1",
    )
    .add_local_dir(str(CODE_DIR), remote_path="/root/code")
)
volume = modal.Volume.from_name("atdl-double-descent-vision-pilot", create_if_missing=True)
