from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def downloader() -> ModuleType:
    path = Path(__file__).parents[1] / 'scripts/generation/download_objaverse.py'
    spec = importlib.util.spec_from_file_location('download_objaverse', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixtures(root: Path) -> tuple[Path, Path, bytes]:
    content = b'original textured asset'
    manifest = root / 'attribution.json'
    manifest.write_text(json.dumps({'assets': [
        {'uid': uid, 'sha256': hashlib.sha256(content).hexdigest(),
         'license_spdx': 'CC-BY-4.0', 'author': 'Artist'} for uid in ['b', 'a']]}))
    paths = root / 'paths.json.gz'
    with gzip.open(paths, 'wt') as stream:
        json.dump({uid: f'glbs/000/{uid}.glb' for uid in ['a', 'b']}, stream)
    return manifest, paths, content


def test_exact_manifest_reuse_repair_and_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = downloader()
    manifest, paths, content = fixtures(tmp_path)
    out = tmp_path / 'assets'
    (out / 'glbs').mkdir(parents=True)
    (out / 'glbs/a.glb').write_bytes(content)
    (out / 'glbs/b.glb').write_bytes(b'corrupt cache')
    calls: list[str] = []

    def fetch(url: str, dest: Path) -> None:
        calls.append(url)
        dest.write_bytes(content)

    monkeypatch.setattr(module, 'fetch', fetch)
    assert module.run(out=out, manifest=manifest, object_paths=paths, workers=2) == 0
    assert len(calls) == 1 and calls[0].endswith('/b.glb')
    assert next((out / 'glbs').glob('b.glb.preserved-*')).read_bytes() == b'corrupt cache'
    index = json.loads((out / 'index.json').read_text())
    assert index['selection'] == ['b', 'a']
    assert index['downloaded'] == ['b', 'a']
    assert index['objects']['a']['license'] == 'CC-BY-4.0'
    assert index['objects']['a']['aesthetic_score'] is None
    complete = json.loads((out / 'complete.json').read_text())
    assert complete['count'] == 2 and complete['verified_bytes'] == 2 * len(content)
    assert module.run(out=out, manifest=manifest, object_paths=paths) == 0
    assert len(calls) == 1
    assert list(out.glob('complete.json.preserved-*'))


def test_failure_excludes_unverified_and_removes_stale_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = downloader()
    manifest, paths, content = fixtures(tmp_path)
    out = tmp_path / 'assets'
    out.mkdir()
    (out / 'complete.json').write_text('{"count":2}')

    def fetch(url: str, dest: Path) -> None:
        dest.write_bytes(b'wrong upstream bytes')

    monkeypatch.setattr(module, 'fetch', fetch)
    assert module.run(out=out, manifest=manifest, object_paths=paths) == 1
    assert not (out / 'complete.json').exists()
    index = json.loads((out / 'index.json').read_text())
    assert index['num_downloaded'] == 0 and len(index['failed']) == 2
    assert json.loads((out / 'status.json').read_text())['phase'] == 'failed'
    monkeypatch.setattr(module.fire, 'Fire', lambda function: 1)
    with pytest.raises(SystemExit, match='1'):
        module.main()


def test_manifest_rejects_duplicate_or_missing_mapping(tmp_path: Path) -> None:
    module = downloader()
    manifest, paths, content = fixtures(tmp_path)
    with pytest.raises(ValueError, match='absent'):
        module.manifest_entries(manifest, {}, None)
    assets = json.loads(manifest.read_text())
    assets['assets'].append(assets['assets'][0])
    manifest.write_text(json.dumps(assets))
    with pytest.raises(ValueError, match='duplicate'):
        module.manifest_entries(manifest, {'a': 'a', 'b': 'b'}, None)


def test_http_partial_transfer_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = downloader()
    target = tmp_path / 'asset.glb'
    target.with_suffix('.glb.part').write_bytes(b'abc')

    class Response:
        status_code = 206
        headers = {'Content-Length': '3'}

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def iter_content(self, chunk_size: int) -> list[bytes]:
            return [b'def']

    def get(url: str, **kwargs: object) -> Response:
        assert kwargs['headers'] == {'Range': 'bytes=3-'}
        return Response()

    monkeypatch.setattr(module.requests, 'get', get)
    module.fetch('https://example.test/asset.glb', target)
    assert target.read_bytes() == b'abcdef'


def test_seeded_csv_selection_still_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = downloader()
    manifest, paths, content = fixtures(tmp_path)
    out = tmp_path / 'assets'
    out.mkdir()
    digest = hashlib.sha256(content).hexdigest()
    (out / 'ObjaverseXL_sketchfab.csv').write_text(
        'sha256,file_identifier,aesthetic_score,captions\n'
        f'{digest},https://sketchfab.com/a,6.0,caption\n'
        f'{digest},https://sketchfab.com/b,2.0,excluded\n')

    def metadata(chunk: str, root: Path) -> dict[str, dict]:
        return {'a': {'license': 'by', 'name': 'Original name', 'author': 'Artist'}}

    def fetch(url: str, dest: Path) -> None:
        dest.write_bytes(content)

    monkeypatch.setattr(module, 'chunk_metadata', metadata)
    monkeypatch.setattr(module, 'fetch', fetch)
    assert module.run(out=out, object_paths=paths, limit=1, seed=4) == 0
    index = json.loads((out / 'index.json').read_text())
    assert index['selection'] == ['a']
    assert index['objects']['a']['aesthetic_score'] == 6.0
    assert index['objects']['a']['captions'] == ['caption']
