"""Resume official OmniObject3D archives from the public OpenDataLab mirror.

Keep the downloaded archives, verify their published SHA256, and extract without
overwriting existing files. Only captured videos and raw scans are requested.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import fcntl
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tarfile
import time
from typing import Any

import fire
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = 'https://opendatalab.com/datasets/api/v3/datasets/6639/r/main'


def request_json(method: str, **options: Any) -> dict[str, Any]:
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods={'GET', 'POST'})
    with requests.Session() as session:
        session.mount('https://', HTTPAdapter(max_retries=retry))
        response = session.request(method, API, timeout=60, **options)
        response.raise_for_status()
        return response.json()


def completed_transfer_bytes(part: Path, expected_size: int) -> int:
    """Count aria2's saved complete pieces; network filesystems can count holes
    as allocated blocks, making both file size and st_blocks misleading.
    The v1 network-order format is defined in aria2 DefaultBtProgressInfoFile.
    In-flight pieces are excluded, so this is a conservative byte count.
    """
    control = part.with_name(part.name + '.aria2')
    try:
        data = control.read_bytes()
    except FileNotFoundError:
        # The control file is removed atomically when its download finishes.
        return 0
    if data[:2] != b'\x00\x01':
        raise ValueError(f'Unsupported aria2 control version: {control}')
    hash_length = struct.unpack_from('!I', data, 6)[0]
    offset = 10 + hash_length
    piece_size, total, _, field_size = struct.unpack_from('!IQQI', data, offset)
    if total != expected_size or piece_size == 0:
        raise ValueError(f'Unexpected aria2 transfer metadata: {control}')
    field = data[offset + 24:offset + 24 + field_size]
    complete = sum(byte.bit_count() for byte in field) * piece_size
    last_piece = (total - 1) // piece_size
    if field[last_piece // 8] & (1 << (7 - last_piece % 8)):
        complete -= (piece_size - total % piece_size) % piece_size
    return min(total, complete)


def progress(root: Path, rows: list[dict[str, Any]], started: float,
             initial_bytes: int = 0) -> dict[str, Any]:
    downloaded = verified = completed = 0
    for row in rows:
        archive = root / 'archives' / Path(row['path']).relative_to('/raw')
        marker = archive.with_suffix(archive.suffix + '.extracted.json')
        if marker.exists():
            downloaded += row['size']
            verified += row['size']
            completed += 1
        elif archive.exists():
            downloaded += min(row['size'], archive.stat().st_size)
        else:
            part = archive.with_suffix(archive.suffix + '.part')
            downloaded += completed_transfer_bytes(part, row['size'])
    elapsed = max(time.monotonic() - started, .001)
    rate = max(0, downloaded - initial_bytes) / elapsed
    total = sum(row['size'] for row in rows)
    return {'archives_complete': completed, 'archives_total': len(rows),
            'downloaded_bytes': downloaded, 'verified_extracted_bytes': verified,
            'total_bytes': total, 'elapsed_seconds': elapsed, 'bytes_per_second': rate,
            'download_eta_hours': (total - downloaded) / rate / 3600 if rate else None}


def save_progress(root: Path, payload: dict[str, Any]) -> None:
    path = root / 'progress.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def catalogue(modality: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    after = ''
    while True:
        payload = request_json('GET', params={'prefix': f'/raw/{modality}/', 'after': after, 'limit': 1000})
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
    archive = root / 'archives' / Path(row['path']).relative_to('/raw')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.with_suffix(archive.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return fetch_archive_contents(row, root)


def fetch_archive_contents(row: dict[str, Any], root: Path) -> dict[str, Any]:
    relative = Path(row['path']).relative_to('/raw')
    archive = root / 'archives' / relative
    destination = root / relative.parent
    marker = archive.with_suffix(archive.suffix + '.extracted.json')
    if marker.exists():
        return {'path': str(relative), 'status': 'already_complete'}
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        payload = request_json('POST', json={'path': row['path'].lstrip('/'), 'preview': False})
        if payload['code'] != 0:
            raise RuntimeError(f'OpenDataLab URL: {payload["msg"]}')
        part = archive.with_suffix(archive.suffix + '.part')
        # URL is supplied over stdin so signed query strings never appear in logs.
        result = subprocess.run(['aria2c', '--continue=true', '--split=12', '--max-connection-per-server=12',
                                 '--min-split-size=8M', '--max-tries=8', '--retry-wait=5',
                                 '--connect-timeout=60', '--file-allocation=none',
                                 '--auto-file-renaming=false', '--summary-interval=0',
                                 '--auto-save-interval=10',
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


def create_plan(root: Path, destination: Path, shards: int) -> dict[str, Any]:
    if shards < 1 or destination.exists():
        raise ValueError('Use a fresh plan path and a positive shard count')
    rows = json.loads((root / 'catalogue.json').read_text())['files']
    bins: list[list[dict[str, Any]]] = [[] for _ in range(shards)]
    loads = [0] * shards
    remaining: dict[str, int] = {}
    for row in rows:
        received = progress(root, [row], time.monotonic())['downloaded_bytes']
        remaining[row['path']] = row['size'] - received
    for row in sorted(rows, key=lambda row: (-remaining[row['path']], row['path'])):
        index = min(range(shards), key=lambda i: loads[i])
        bins[index].append(row)
        loads[index] += remaining[row['path']]
    identities = [row['path'] for group in bins for row in group]
    if len(set(identities)) != len(rows):
        raise ValueError('Archive ownership must be unique across shards')
    result = {'schema': 1, 'source': API, 'shards': bins, 'remaining_bytes_per_shard': loads,
              'total_bytes': sum(row['size'] for row in rows), 'archives': len(rows)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    return {'shards': shards, 'archives': len(rows), 'remaining_bytes_per_shard': loads}


def main(out: str, workers: int = 3, category: str = '', measure_seconds: int = 0,
         plan: str = '', shard: int = -1, create_shards: int = 0) -> None:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    if create_shards:
        print(json.dumps(create_plan(root, Path(plan), create_shards), indent=2))
        return
    if measure_seconds:
        rows = json.loads((root / 'catalogue.json').read_text())['files']
        started = time.monotonic()
        initial_bytes = progress(root, rows, started)['downloaded_bytes']
        time.sleep(measure_seconds)
        report = progress(root, rows, started, initial_bytes)
        report['progress_method'] = 'aria2_completed_pieces'
        report['measured_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        (root / 'measured-progress.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return
    progress_root = root
    if plan:
        groups = json.loads(Path(plan).read_text())['shards']
        if not 0 <= shard < len(groups):
            raise ValueError('Invalid shard index')
        rows = groups[shard]
        progress_root = Path(plan).parent / f'shard-{shard:02d}'
        progress_root.mkdir(parents=True, exist_ok=True)
    else:
        rows = [row for modality in ['raw_scans', 'videos_processed'] for row in catalogue(modality)]
    if category:
        rows = [row for row in rows if Path(row['path']).name == category + '.tar.gz']
    if not rows:
        raise ValueError('No source archives selected')
    if not plan:
        (root / 'catalogue.json').write_text(json.dumps({'source': API, 'files': rows,
            'bytes': sum(row['size'] for row in rows)}, indent=2))
    print(json.dumps({'archives': len(rows), 'bytes': sum(row['size'] for row in rows)}), flush=True)
    failures: dict[str, str] = {}
    started = time.monotonic()
    initial_bytes = progress(root, rows, started)['downloaded_bytes']
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_archive, row, root): row['path'] for row in rows}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                error = future.exception()
                if error is None:
                    print(json.dumps(future.result()), flush=True)
                else:
                    failures[futures[future]] = str(error)
                    print(json.dumps({'failed': futures[future], 'error': str(error)}), flush=True)
            report = progress(root, rows, started, initial_bytes)
            report['failed_archives'] = len(failures)
            save_progress(progress_root, report)
            print(json.dumps({'progress': report}), flush=True)
    (progress_root / 'download_failures.json').write_text(json.dumps(failures, indent=2))
    if failures:
        raise RuntimeError(f'{len(failures)} archives failed; rerun resumes completed work')


if __name__ == '__main__':
    fire.Fire(main)
