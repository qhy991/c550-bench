"""Create-only request admission and complete candidate snapshots."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import stat

from .owner import publish_new

NAME = re.compile(r'[a-z0-9][a-z0-9_-]{0,63}')
MAX_SOURCE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class Request:
    identity: str
    phase: str
    candidate: str

    @classmethod
    def parse(cls, value: dict, folder: str):
        if (set(value) != {'id', 'phase', 'candidate'}
                or not isinstance(value['id'], str) or not NAME.fullmatch(value['id'])
                or value['id'] != folder or value['phase'] not in {'search', 'profile'}
                or not isinstance(value['candidate'], str)
                or not NAME.fullmatch(value['candidate'])):
            raise ValueError('request must name one own candidate and search or profile')
        return cls(value['id'], value['phase'], value['candidate'])


def snapshot(candidate_root: Path, identity: str, destination: Path) -> Path:
    """Copy one complete source bundle without following author-controlled links."""
    if not NAME.fullmatch(identity):
        raise ValueError('invalid candidate identity')
    source = candidate_root / identity
    if source.is_symlink() or not source.is_dir() or source.resolve() != source:
        raise ValueError('candidate must be one real directory under candidates')
    destination.mkdir(parents=True, exist_ok=False)
    total = 0

    def copy_directory(fd, output, depth=0):
        nonlocal total
        if depth > 16:
            raise ValueError('candidate directory nesting exceeds limit')
        for name in os.listdir(fd):
            metadata = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                try:
                    (output / name).mkdir()
                    copy_directory(child, output / name, depth + 1)
                finally:
                    os.close(child)
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValueError('candidate entries must be regular unlinked source or directories')
            if Path(name).suffix not in {'.py', '.json'}:
                raise ValueError('candidate bundle accepts Python and JSON source only')
            child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
            try:
                before = os.fstat(child)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                    raise ValueError('candidate file changed kind during snapshot')
                total += before.st_size
                if total > MAX_SOURCE_BYTES:
                    raise ValueError('candidate bundle exceeds source limit')
                with os.fdopen(child, 'rb', closefd=False) as stream:
                    contents = stream.read(MAX_SOURCE_BYTES + 1)
                after = os.fstat(child)
                if (len(contents) != before.st_size or (before.st_mtime_ns, before.st_ctime_ns) !=
                        (after.st_mtime_ns, after.st_ctime_ns)):
                    raise ValueError('candidate changed during snapshot')
                (output / name).write_bytes(contents)
            finally:
                os.close(child)

    fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        copy_directory(fd, destination)
    finally:
        os.close(fd)
    if not (destination / 'candidate.py').is_file():
        raise ValueError('candidate bundle must expose candidate.py run(*inputs)')
    for path in destination.rglob('*'):
        path.chmod(0o555 if path.is_dir() else 0o444)
    destination.chmod(0o555)
    return destination


def read_request(path: Path) -> dict:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 4096:
            raise ValueError('request must be a bounded regular JSON file')
        with os.fdopen(fd, 'r', closefd=False) as stream:
            value = json.loads(stream.read(4097))
        if not isinstance(value, dict):
            raise ValueError('request must be a JSON object')
        return value
    finally:
        os.close(fd)


def nominate(evaluations: list[dict], *, search_stop: float) -> dict | None:
    profiles = {row['candidate'] for row in evaluations
                if row.get('phase') == 'profile' and row.get('status') == 'passed'
                and row.get('completed_at_epoch', float('inf')) <= search_stop}
    rows = [row for row in evaluations if row.get('phase') == 'search'
            and row.get('status') == 'passed' and row.get('candidate') in profiles
            and row.get('completed_at_epoch', float('inf')) <= search_stop
            and row.get('measurement', {}).get('quality_passed') is True]
    if not rows:
        return None
    winner = min(rows, key=lambda row: (row['measurement']['candidate_over_baseline'], row['id']))
    return {'id': winner['candidate'], 'status': 'eligible',
            'completed_at_epoch': max(row['completed_at_epoch'] for row in evaluations
                                     if row['candidate'] == winner['candidate']),
            'search_evaluation': winner['id']}


def submit(root: Path, identity: str, phase: str, candidate: str) -> dict:
    """Author client. A missing response consumes this request and is never retried."""
    import time
    request = {'id': identity, 'phase': phase, 'candidate': candidate}
    Request.parse(request, identity)
    deadline = json.loads((root / 'campaign/deadline.json').read_text())
    if time.time() >= deadline['search_stop_at_epoch']:
        raise TimeoutError('search deadline is closed')
    publish_new(root / 'campaign/requests' / f'{identity}.json', request)
    response = root / 'campaign/responses' / f'{identity}.json'
    while time.time() < deadline['search_stop_at_epoch']:
        if response.exists():
            return json.loads(response.read_text())
        time.sleep(0.2)
    raise TimeoutError('request response unknown; this request must not be resubmitted')
