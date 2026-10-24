"""Launching BlenderProc scripts from the uv environment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BLENDER_SCRIPTS = Path(__file__).parent / "blender"


def run_blenderproc(script: str, *args: str, install_path: str, log_path: Path | None = None) -> None:
    """Run `blenderproc run <script> <args>`; raises CalledProcessError with the log path on failure."""
    exe = Path(sys.executable).parent / "blenderproc"
    cmd = [str(exe), "run", "--blender-install-path", install_path, str(BLENDER_SCRIPTS / script), *args]
    if log_path is None:
        subprocess.run(cmd, check=True)
        return
    with open(log_path, "w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, output=f"see {log_path}")
