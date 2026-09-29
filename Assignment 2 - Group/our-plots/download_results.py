"""Download the authors' published result objects from a pinned manifest."""

from __future__ import annotations

import base64
import hashlib
import shutil
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import fire
from pydantic import BaseModel


class ResultObject(BaseModel):
    name: str
    size: int
    md5_hash: str
    generation: str


class ResultManifest(BaseModel):
    bucket: str
    source_commit: str
    objects: list[ResultObject]


def fetch(manifest_path: str | None = None, destination: str | None = None) -> None:
    plot_dir = Path(__file__).resolve().parent
    assignment_dir = plot_dir.parent
    manifest_file = Path(manifest_path) if manifest_path else assignment_dir / "their-results" / "manifest.json"
    output_dir = Path(destination) if destination else assignment_dir / "their-results"
    manifest = ResultManifest.model_validate_json(manifest_file.read_text())

    for item in manifest.objects:
        target = output_dir / item.name
        if target.exists() and target.stat().st_size == item.size:
            digest = md5_digest(target)
            if base64.b64encode(digest).decode() == item.md5_hash:
                continue

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".download")
        url = f"https://storage.googleapis.com/{manifest.bucket}/{quote(item.name, safe='/')}?generation={item.generation}"
        with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if temporary.stat().st_size != item.size:
            raise ValueError(f"Size mismatch for {item.name}")
        digest = md5_digest(temporary)
        if base64.b64encode(digest).decode() != item.md5_hash:
            raise ValueError(f"Hash mismatch for {item.name}")
        temporary.replace(target)
        print(f"Downloaded {item.name}")

    print(f"Verified {len(manifest.objects)} result files in {output_dir}")


def md5_digest(path: Path) -> bytes:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


if __name__ == "__main__":
    fire.Fire(fetch)
