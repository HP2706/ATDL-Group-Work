"""Release allowlist tests: python tests/test_objaverse_licenses.py."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("download_objaverse", Path(__file__).parents[1] / "scripts/download_objaverse.py")
obj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(obj)


class TestObjectLicenses(unittest.TestCase):
    def test_allowlist_and_attribution_fail_closed(self):
        base = {"name": "Cup", "author": "Artist", "source_url": "https://sketchfab.com/models/a"}
        entries = [{**base, "uid": str(i), "license": code} for i, code in enumerate(
            ["by", "cc0", "by-sa", "by-nc", "by-nd", "by-nc-sa", "by-nc-nd", None, "unknown"])]
        entries.append({**base, "uid": "missing", "license": "by", "author": None})
        good, rejected = obj.release_entries(entries)
        self.assertEqual([x["license_spdx"] for x in good], ["CC-BY-4.0", "CC0-1.0"])
        self.assertEqual(len(rejected), 8)
        self.assertEqual(rejected[-1]["reason"], "missing_attribution")
        self.assertTrue(all(x["license_url"].startswith("https://creativecommons.org/") for x in good))

    def test_index_and_credits_only_reference_allowed_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "glbs").mkdir()
            (root / "glbs/a.glb").write_bytes(b"mesh")
            entries, _ = obj.release_entries([
                {"uid": "a", "license": "by", "author": "Artist", "name": "Cup", "source_url": "https://example.org/a"},
                {"uid": "b", "license": "cc0"}])
            args = argparse.Namespace(min_aesthetic=5.5, seed=0, limit=10, license_rejections=[])
            self.assertEqual(obj.write_index(root, args, 20, entries, {}), 1)
            index = json.loads((root / "index.json").read_text())
            credits = json.loads((root / "ATTRIBUTION.json").read_text())
            self.assertEqual(index["selection"], ["a", "b"])
            self.assertEqual([x["uid"] for x in credits["assets"]], ["a"])
            self.assertEqual(credits["assets"][0]["author"], "Artist")

    def test_corrupt_existing_glb_is_downloaded_again(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "glbs").mkdir()
            dest = root / "glbs/a.glb"
            dest.write_bytes(b"bad")
            entry = {"uid": "a", "sha256": hashlib.sha256(b"good").hexdigest()}
            with patch.object(obj, "fetch", side_effect=lambda url, path: path.write_bytes(b"good")) as fetch:
                self.assertTrue(obj.download_glb(entry, {"a": "glbs/000/a.glb"}, root))
                fetch.assert_called_once()
                self.assertFalse(obj.download_glb(entry, {"a": "glbs/000/a.glb"}, root))


if __name__ == "__main__":
    unittest.main()
