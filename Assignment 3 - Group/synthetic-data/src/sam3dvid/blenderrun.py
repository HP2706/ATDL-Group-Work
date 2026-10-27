"""Launching BlenderProc scripts from the uv environment."""

from __future__ import annotations

import collections
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

BLENDER_SCRIPTS = Path(__file__).parent / "blender"


def run_blenderproc(
    script: str, *args: str, install_path: str, custom_path: str | None = None, log_path: Path | None = None,
    stall_timeout: float | None = None,
) -> None:
    """Run `blenderproc run <script> <args>`; raises CalledProcessError with the log path on failure.

    With `custom_path`, that Blender installation is used instead of BlenderProc's own 4.2.1 in
    `install_path`. With `stall_timeout`, the process group is killed after that many seconds without
    any output and TimeoutExpired is raised; only lines starting with "[" are echoed to stdout.
    """
    exe = Path(sys.executable).parent / "blenderproc"
    blender = ["--custom-blender-path", custom_path] if custom_path else ["--blender-install-path", install_path]
    cmd = [str(exe), "run", *blender, str(BLENDER_SCRIPTS / script), *args]
    if stall_timeout is not None:
        run_with_stall_timeout(cmd, stall_timeout)
        return
    if log_path is None:
        subprocess.run(cmd, check=True)
        return
    with open(log_path, "w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, output=f"see {log_path}")


def run_with_stall_timeout(cmd: list[str], stall_timeout: float) -> None:
    """Run cmd, echoing its "[..." lines; kill its process group after `stall_timeout` s of silence."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True
    )
    last_output = time.monotonic()
    tail: collections.deque[str] = collections.deque(maxlen=30)

    def pump() -> None:
        nonlocal last_output
        for line in proc.stdout:
            last_output = time.monotonic()
            tail.append(line)
            if line.startswith("["):
                print(line, end="", flush=True)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    while proc.poll() is None:
        if time.monotonic() - last_output > stall_timeout:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            reader.join()
            raise subprocess.TimeoutExpired(cmd, stall_timeout)
        time.sleep(1)
    reader.join()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output="".join(tail))
