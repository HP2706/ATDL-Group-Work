"""CPU-only tests; also runnable without the renderer: python tests/test_polyhaven.py."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location("prepare_polyhaven", Path(__file__).parents[2] / "scripts/generation/prepare_polyhaven.py")
ph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ph)


class TestPolyHaven(unittest.TestCase):
    def test_projection_preserves_constant_panorama(self):
        image = np.full((64, 128, 3), (24, 80, 160), np.uint8)
        for yaw in (0, 90, 180, 359):
            out = ph.perspective(image, yaw, 41, 21, 130)
            np.testing.assert_array_equal(out, np.broadcast_to(image[0, 0], out.shape))

    def test_projection_direction_and_seam(self):
        image = np.zeros((180, 360, 3), np.uint8)
        image[:, :, 0] = np.arange(360)[None, :] % 256
        image[:, :, 1] = np.arange(180)[:, None]
        for yaw in (0, 90, 180):
            out = ph.perspective(image, yaw, 5, 3, 90)
            center_u = ((yaw / 360 + 0.5) * 360 - 0.5) % 360
            left = int(np.floor(center_u))
            expected = round((int(image[90, left, 0]) + int(image[90, (left + 1) % 360, 0])) / 2)
            self.assertLessEqual(abs(int(out[1, 2, 0]) - expected), 1)
            self.assertLessEqual(abs(int(out[1, 2, 1]) - 90), 1)

    def test_download_rejects_corrupt_content(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def raise_for_status(self): pass
            def iter_content(self, size): yield b"corrupt"
        class Session:
            def get(self, *args, **kwargs): return Response()
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "pano.jpg"
            with self.assertRaisesRegex(ValueError, "checksum"):
                ph.download(Session(), {"url": "https://example.invalid/a", "md5": "bad"}, dest)
            self.assertFalse(dest.exists())

    def test_uppercase_api_checksum_matches_cached_download(self):
        class NoNetwork:
            def get(self, *args, **kwargs):
                raise AssertionError("Valid cached image should not be downloaded again")
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "pano.jpg"
            dest.write_bytes(b"image")
            checksum = hashlib.md5(b"image").hexdigest().upper()
            ph.download(NoNetwork(), {"md5": checksum}, dest)

    def test_prepare_resume_provenance_and_grouped_splits(self):
        ids = [f"asset_{i}" for i in range(100)]
        train = next(x for x in ids if ph.source_split(x, 0.5) == "train")
        evaluation = next(x for x in ids if ph.source_split(x, 0.5) == "eval")
        catalog = {x: {"authors": {"Example": "Author"}} for x in (train, evaluation)}
        buf = io.BytesIO()
        Image.new("RGB", (128, 64), (80, 100, 120)).save(buf, format="JPEG")
        content = buf.getvalue()
        checksum = hashlib.md5(content).hexdigest()
        calls = []
        class Response:
            def __init__(self, value=None): self.value = value
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def raise_for_status(self): pass
            def json(self): return self.value
            def iter_content(self, size): yield content
        class Session:
            headers = {}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def mount(self, *args): pass
            def get(self, url, **kwargs):
                calls.append(url)
                if url.endswith("/assets"): return Response(catalog)
                if "/files/" in url:
                    return Response({"tonemapped": {"url": "https://example.invalid/pano.jpg", "md5": checksum}})
                return Response()
        with tempfile.TemporaryDirectory() as directory, patch.object(ph.requests, "Session", Session):
            args = argparse.Namespace(root=Path(directory), raw=Path("raw"), out=Path("pools"),
                                      limit=2, seed=0, views=4, width=32, height=16, hfov=130, eval_frac=0.5)
            stats = ph.prepare(args)
            self.assertEqual(stats["pools"], {"train": 4, "eval": 4})
            manifest = json.loads((args.root / "pools/manifest.json").read_text())
            records = manifest["images"]
            self.assertEqual({x["license"] for x in records}, {"CC0-1.0"})
            sources = {split: {r["source_id"] for r in records if r["split"] == split} for split in ("train", "eval")}
            self.assertFalse(sources["train"] & sources["eval"])
            for record in records:
                self.assertEqual(ph.digest(args.root / record["path"]), record["sha256"])
            calls.clear()
            self.assertEqual(ph.prepare(args), stats)
            self.assertEqual(calls, [])
            args.views = 8
            with self.assertRaisesRegex(ValueError, "settings changed"):
                ph.prepare(args)


if __name__ == "__main__":
    unittest.main()
