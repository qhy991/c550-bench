"""Confirm one fixed, source-bound nominee inside the original total budget on C500."""
import copy
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _source_bound(root, outcome):
    bindings = dict(outcome.get('cake_artifact_bindings', {}))
    bindings[outcome['candidate_source']] = outcome['source_sha256']
    return all(
        (root / name).is_file()
        and hashlib.sha256((root / name).read_bytes()).hexdigest() == value
        for name, value in bindings.items()
    )


def _acquire_device_lock(device):
    uid = os.geteuid()
    lock = Path(f'/tmp/open-cake-ir-maca-{uid}-device-{device}.lock')
    fd = os.open(str(lock), os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _release_device_lock(fd):
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def main():
    os.chdir(ROOT)
    deadline = json.loads((ROOT / 'campaign/deadline.json').read_text())
    binding = json.loads((ROOT / 'campaign/binding.json').read_text())
    device = binding['physical_device']
    task = binding['task']
    baseline = binding['baseline']

    folder = ROOT / 'campaign/final-confirmation'
    folder.mkdir(exist_ok=False)

    outcomes = [
        json.loads(p.read_text())
        for p in (ROOT / 'campaign/evaluations').glob('*/outcome.json')
    ]
    eligible = [
        x for x in outcomes
        if x.get('status') == 'accepted'
        and x.get('completed_at_epoch', 0) <= deadline['search_stop_at_epoch']
    ]
    stale = [x['id'] for x in eligible if not _source_bound(ROOT, x)]
    eligible = [x for x in eligible if x['id'] not in stale]
    nominee = max(eligible, key=lambda x: x['conservative_geomean']) if eligible else None

    result = {
        'scope': 'one fixed nominee; assay confirmation inside total budget; attribution pending independent review',
        'status': 'missing_qualified_nominee',
        'nominated_id': None,
        'stale_artifacts_excluded': stale,
        'budget_stop_epoch': deadline['stop_at_epoch'],
    }

    try:
        if nominee:
            result['nominated_id'] = nominee['id']
            (folder / 'nomination.json').write_text(json.dumps(nominee, indent=2))
            timeout = min(900, int(deadline['stop_at_epoch'] - time.time()) - 45)
            if timeout < 10:
                result['status'] = 'insufficient_remaining_budget'
            else:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device))
                for k in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY',
                          'all_proxy', 'ALL_PROXY'):
                    env.pop(k, None)

                latency_out = folder / 'latency.json'
                cmd = [
                    sys.executable, 'campaign/paired_wall.py',
                    '--task', task,
                    '--baseline', baseline,
                    '--candidate', nominee['candidate_source'],
                    '--gate', nominee['handoff_row']['correctness_report'],
                    '--samples', '30',
                    '--output', str(latency_out.relative_to(ROOT)),
                ]
                lock_fd = _acquire_device_lock(device)
                try:
                    with (folder / 'execution.log').open('x') as log:
                        code = subprocess.call(cmd, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, timeout=timeout)
                finally:
                    _release_device_lock(lock_fd)

                result.update(status='confirmation_failed', exit_code=code)
                if code == 0:
                    if not _source_bound(ROOT, nominee):
                        raise ValueError('nominee artifacts changed during final confirmation')
                    report = json.loads(latency_out.read_text())
                    completed_at = time.time()
                    timely = completed_at <= deadline['stop_at_epoch']
                    ratios = [
                        min(c['baseline_us'][order] / c['candidate_us'][order]
                            for order in ['forward_median', 'reverse_median'])
                        for c in report['rows']
                    ]
                    result.update(
                        status='confirmed' if timely else 'not_confirmed',
                        conservative_geomean=math.exp(
                            sum(math.log(x) for x in ratios) / len(ratios)
                        ),
                        rows=len(ratios),
                        released=True,
                        completed_within_budget=timely,
                    )
    except (KeyError, ValueError, OSError, subprocess.SubprocessError) as error:
        result.update(status='confirmation_failed',
                      error=type(error).__name__ + ': ' + str(error))

    result['completed_node_epoch'] = time.time()
    with (folder / 'result.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(result, flush=True)


if __name__ == '__main__':
    main()
