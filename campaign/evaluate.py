"""Freeze and evaluate one candidate through the correctness/wall path on C500."""
import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def inside(name):
    path = (ROOT / name).resolve()
    path.relative_to(ROOT.resolve())
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _acquire_device_lock(device):
    uid = os.geteuid()
    lock = Path(f'/tmp/open-cake-ir-maca-{uid}-device-{device}.lock')
    fd = os.open(str(lock), os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _release_device_lock(fd):
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _relative(path):
    return str(path.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--cake-artifact', action='append', default=[])
    parser.add_argument('--profile-skip-reason', default='profile_unavailable_on_c500')
    args = parser.parse_args()

    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', args.id):
        parser.error('id must be a unique lowercase artifact name')

    # B1: every candidate in a Compiler comparison bench must be a Cake emission
    if not args.cake_artifact:
        parser.error(
            'all candidates require --cake-artifact receipts; '
            'Native Torch/Triton/HIP or torch.compile is not a counted Cake solution'
        )

    os.chdir(str(ROOT))
    binding = json.loads((ROOT / 'campaign/binding.json').read_text())
    plan = json.loads((ROOT / 'campaign/frozen-plan.json').read_text())
    deadline = json.loads((ROOT / 'campaign/deadline.json').read_text())

    task = binding['task']
    device = binding['physical_device']
    baseline = binding['baseline']
    rounds = binding['correctness_rounds']

    folder = ROOT / 'campaign/evaluations' / args.id
    folder.mkdir(parents=True, exist_ok=False)

    outcome = {
        'id': args.id, 'task': task, 'started_at_epoch': time.time(),
        'status': 'not_qualified', 'profile_skip_reason': args.profile_skip_reason,
        'cake_artifacts': args.cake_artifact,
    }

    try:
        if time.time() >= deadline['search_stop_at_epoch']:
            raise ValueError('search budget already expired')

        frozen = ROOT / 'campaign/candidates' / ('frozen_' + args.id + '.py')
        with frozen.open('x') as stream:
            stream.write(inside(args.candidate).read_text())
        source_hash = digest(frozen)

        artifact_bindings = {}
        for name in args.cake_artifact:
            record = json.loads(inside(name).read_text())
            for path in (name, record['source'], record['schedule']):
                artifact_bindings[path] = digest(inside(path))
        subprocess.check_call([
            sys.executable, 'campaign/cake_bridge.py', 'verify-many',
        ] + args.cake_artifact, timeout=180)

        # B3: precision guard (CPU-only, before device work)
        from campaign.precision_guard import validate as precision_validate
        precision = precision_validate(ROOT, task, args.cake_artifact)
        (folder / 'precision.json').write_text(json.dumps(precision, indent=2) + '\n')

        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device))
        for k in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY',
                  'http_proxy', 'https_proxy', 'all_proxy', 'ALL_PROXY'):
            env.pop(k, None)

        gate = folder / 'correctness.json'
        caller_out = folder / 'caller-guards.json'
        latency = folder / 'latency.json'

        lock_fd = _acquire_device_lock(device)
        try:
            code = subprocess.call([
                sys.executable, 'c550bench.py', 'check',
                '--task', task, '--device', 'cuda:0',
                '--candidate', _relative(frozen),
                '--workloads', 'all', '--rounds', str(rounds),
                '--output', _relative(gate),
            ], env=env)
            if code:
                raise ValueError('correctness check failed: exit ' + str(code))

            gate_record = json.loads(gate.read_text())
            if not gate_record.get('full_device_correctness') or gate_record.get('status') != 'passed':
                raise ValueError('candidate failed full device correctness: ' + json.dumps({
                    k: gate_record.get(k) for k in ('full_device_correctness', 'status')}))

            # B3: caller guard (input-ownership and output-retention checks)
            code = subprocess.call([
                sys.executable, 'campaign/caller_guards.py',
                '--candidate', _relative(frozen),
                '--output', _relative(caller_out),
            ], env=env)
            if code:
                raise ValueError('caller guard failed: exit ' + str(code))

            code = subprocess.call([
                sys.executable, 'campaign/paired_wall.py',
                '--task', task,
                '--baseline', baseline,
                '--candidate', _relative(frozen),
                '--gate', _relative(gate),
                '--output', _relative(latency),
            ], env=env)
            if code:
                raise ValueError('latency measurement failed: exit ' + str(code))
        finally:
            _release_device_lock(lock_fd)

        if digest(frozen) != source_hash:
            raise ValueError('frozen candidate changed during evaluation')
        if any(digest(inside(name)) != expected for name, expected in artifact_bindings.items()):
            raise ValueError('Compiler artifact changed during evaluation')
        if time.time() > deadline['search_stop_at_epoch']:
            raise ValueError('evaluation finished after the search budget')

        timing = json.loads(latency.read_text())
        ratios = [min(c['baseline_us'][k] / c['candidate_us'][k]
                      for k in ('forward_median', 'reverse_median'))
                  for c in timing['rows']]
        handoff_row = {
            'task': task, 'status': 'accepted',
            'reason': 'full device correctness gate, caller guard, and paired wall timing on C500',
            'evidence': [_relative(gate), _relative(caller_out), _relative(latency)],
            'candidate_source': _relative(frozen),
            'correctness_source_sha256': source_hash,
            'correctness_report': _relative(gate),
            'caller_report': _relative(caller_out),
            'precision_report': _relative(folder / 'precision.json'),
            'latency_report': _relative(latency),
            'profile_skip_reason': args.profile_skip_reason,
            'cake_artifacts': args.cake_artifact,
        }
        outcome.update(
            status='accepted',
            conservative_geomean=math.exp(sum(map(math.log, ratios)) / len(ratios)),
            speedup_min=min(ratios), speedup_max=max(ratios),
            candidate_source=_relative(frozen),
            source_sha256=source_hash,
            correctness_report=_relative(gate),
            caller_report=_relative(caller_out),
            latency_report=_relative(latency),
            cake_artifact_bindings=artifact_bindings,
            handoff_row=handoff_row,
        )
    except (KeyError, ValueError, OSError, subprocess.SubprocessError) as error:
        outcome['status'] = 'not_qualified'
        outcome['reason'] = str(error)

    outcome.update(completed_at_epoch=time.time())
    with (folder / 'outcome.json').open('x') as stream:
        json.dump(outcome, stream, indent=2)
        stream.write('\n')
    print(json.dumps(outcome, indent=2))
    return 0 if outcome['status'] == 'accepted' else 1


if __name__ == '__main__':
    sys.exit(main())
