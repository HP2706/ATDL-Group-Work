"""Build Cycles' missing sm_80 binary once instead of JIT-compiling on each A100."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sysconfig
import tempfile

import zstandard


def compile_kernel() -> None:
    library = Path("/opt/blender/blender-4.5.15-linux-x64/4.5/scripts/addons_core/cycles/lib")
    source = library / "kernel_compute_75.ptx.zst"
    ptxas = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cuda_nvcc/bin/ptxas"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ptx = root / "kernel.ptx"
        binary = root / "kernel_sm_80.cubin"
        ptx.write_bytes(zstandard.ZstdDecompressor().decompress(source.read_bytes(), max_output_size=1024**3))
        print("PTX_HEADER", ptx.read_text()[:500], flush=True)
        subprocess.run([str(ptxas), "-O1", "--gpu-name=sm_80", str(ptx), "--output-file", str(binary)], check=True)
        destination = library / "kernel_sm_80.cubin.zst"
        destination.write_bytes(zstandard.ZstdCompressor(level=3).compress(binary.read_bytes()))
        print("A100_BINARY", str(destination), destination.stat().st_size, flush=True)


if __name__ == "__main__":
    compile_kernel()
