"""Resume official OmniObject3D archives from the public OpenDataLab mirror.

Keep the downloaded archives, verify their published SHA256, and extract without
overwriting existing files. Only captured videos and raw scans are requested.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from typing import Any

import fire
import requests

API = 'https://opendatalab.com/datasets/api/v3/datasets/6639/r/main'


def catalogue(modality: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    after = ''
    while True:
        response = requests.get(API, params={'prefix': f'/raw/{modality}/', 'after': after, 'limit': 1000}, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if payload['code'] != 0:
            raise RuntimeError(f'OpenDataLab catalogue: {payload["msg"]}')
        page = payload['data']
        rows.extend(row for row in page['list'] if row['schema'] == 'file')
        if not page['hasNext']:
            return rows
        if after == page['after']:
            raise RuntimeError('Catalogue pagination did not advance')
        after = page['after']


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch_archive(row: dict[str, Any], root: Path) -> dict[str, Any]:
    relative = Path(row['path']).relative_to('/raw')
    archive = root / 'archives' / relative
    destination = root / relative.parent
    marker = archive.with_suffix(archive.suffix + '.extracted.json')
    if marker.exists():
        return {'path': str(relative), 'status': 'already_complete'}
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        response = requests.post(API, json={'path': row['path'].lstrip('/'), 'preview': False}, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if payload['code'] != 0:
            raise RuntimeError(f'OpenDataLab URL: {payload["msg"]}')
        part = archive.with_suffix(archive.suffix + '.part')
        # URL is supplied over stdin so signed query strings never appear in logs.
        result = subprocess.run(['aria2c', '--continue=true', '--split=12', '--max-connection-per-server=12',
                                 '--min-split-size=8M', '--max-tries=8', '--retry-wait=5',
                                 '--connect-timeout=60', '--file-allocation=none',
                                 '--auto-file-renaming=false', '--summary-interval=0',
                                 '--console-log-level=warn', '--download-result=hide',
                                 '--input-file=-'],
                                input=(payload['data']['url'] + '\n  dir=' + str(part.parent)
                                       + '\n  out=' + part.name + '\n'), text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(f'aria2c failed ({result.returncode}) for {relative}')
        if part.stat().st_size != row['size'] or sha256(part) != row['sha256']:
            raise ValueError(f'Size/SHA256 mismatch: {relative}; partial file retained')
        part.replace(archive)
    elif archive.stat().st_size != row['size'] or sha256(archive) != row['sha256']:
        raise ValueError(f'Existing archive differs: {relative}; retained for inspection')
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f'Unsafe archive member: {relative}/{member.name}')
        for member in members:
            target = destination / member.name
            if target.exists():
                if member.isfile() and target.stat().st_size != member.size:
                    raise ValueError(f'Existing extracted file differs: {target}')
                continue
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.extracting')
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f'Cannot read archive member: {member.name}')
            with source, temporary.open('wb') as output:
                shutil.copyfileobj(source, output, length=1 << 20)
            temporary.replace(target)
    report = {'path': str(relative), 'sha256': row['sha256'], 'bytes': row['size'], 'members': len(members)}
    marker.write_text(json.dumps(report, indent=2))
    return report


def main(out: str, workers: int = 3, category: str = '') -> None:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    rows = [row for modality in ['raw_scans', 'videos_processed'] for row in catalogue(modality)]
    if category:
        rows = [row for row in rows if Path(row['path']).name == category + '.tar.gz']
    if not rows:
        raise ValueError('No source archives selected')
    (root / 'catalogue.json').write_text(json.dumps({'source': API, 'files': rows,
        'bytes': sum(row['size'] for row in rows)}, indent=2))
    print(json.dumps({'archives': len(rows), 'bytes': sum(row['size'] for row in rows)}), flush=True)
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_archive, row, root): row['path'] for row in rows}
        for future in as_completed(futures):
            error = future.exception()
            if error is None:
                print(json.dumps(future.result()), flush=True)
            else:
                failures[futures[future]] = str(error)
                print(json.dumps({'failed': futures[future], 'error': str(error)}), flush=True)
    (root / 'download_failures.json').write_text(json.dumps(failures, indent=2))
    if failures:
        raise RuntimeError(f'{len(failures)} archives failed; rerun resumes completed work')


if __name__ == '__main__':
    fire.Fire(main)
