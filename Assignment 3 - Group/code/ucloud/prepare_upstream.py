from __future__ import annotations

import hashlib
import json
from pathlib import Path


def patch_geometry_imports(source: Path) -> dict[str, str]:
    path = source / "sam3d_objects/model/backbone/tdfy_dit/models/__init__.py"
    before = "from .structured_latent_vae import SLatGaussianDecoder\nfrom .structured_latent_flow import SLatFlowModel, SLatFlowModelTdfyWrapper\n"
    after = "import os\nif os.environ.get(\"SAM3D_GEOMETRY_ONLY\") != \"1\":\n    from .structured_latent_vae import SLatGaussianDecoder\n    from .structured_latent_flow import SLatFlowModel, SLatFlowModelTdfyWrapper\n"
    original = path.read_text()
    if before in original:
        path.write_text(original.replace(before, after, 1))
    elif after not in original:
        raise ValueError("Pinned upstream initializer changed; refuse to patch")
    return {"file": str(path.relative_to(source)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "purpose": "Skip only unused sparse refinement imports in geometry-only mode; model/loss implementations unchanged."}


def main() -> None:
    record = patch_geometry_imports(Path("/work/dependencies/sam-3d-objects"))
    Path("/work/results").mkdir(exist_ok=True, parents=True)
    Path("/work/results/upstream-import-patch.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
