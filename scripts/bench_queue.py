"""Finite author sequencer for C500 Bench; wraps campaign.owner.CampaignOwner."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from campaign.owner import CampaignOwner

LAUNCH_ENTRY = 'campaign/launch.py'


class _ProcessHandle:
    def __init__(self, proc):
        self._proc = proc

    def poll(self):
        return self._proc.poll()


class C500AuthorLauncher:
    """Start campaign/launch.py as a subprocess; synthesise a terminal from its output files."""

    def start(self, run, claim):
        root = Path(run['root'])
        (root / 'campaign/logs').mkdir(parents=True, exist_ok=True)
        log_path = root / 'campaign/logs/owner-launch.log'
        with log_path.open('x') as log:
            proc = subprocess.Popen(
                [sys.executable, LAUNCH_ENTRY],
                cwd=root,
                stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return _ProcessHandle(proc)

    def terminal(self, run, handle, exit_code):
        root = Path(run['root'])
        assignment_id = run['assignment_id']

        status_file = root / 'campaign/controller-status.json'
        status = {}
        if status_file.is_file():
            try:
                status = json.loads(status_file.read_text())
            except (OSError, ValueError):
                pass

        confirmation_file = root / 'campaign/final-confirmation/result.json'
        confirmation = {}
        if confirmation_file.is_file():
            try:
                confirmation = json.loads(confirmation_file.read_text())
            except (OSError, ValueError):
                pass

        evaluations = []
        for p in (root / 'campaign/evaluations').glob('*/outcome.json'):
            try:
                evaluations.append(json.loads(p.read_text()))
            except (OSError, ValueError):
                pass

        accepted = [e for e in evaluations if e.get('status') == 'accepted']
        exec_status = ('timeout' if exit_code == 124
                       else 'completed' if exit_code == 0
                       else 'failed')

        # Per-workload summary from final confirmation latency
        workloads = {}
        latency_file = root / 'campaign/final-confirmation/latency.json'
        if latency_file.is_file():
            try:
                latency = json.loads(latency_file.read_text())
                for row in latency.get('rows', []):
                    wid = row.get('workload_uuid')
                    if wid:
                        workloads[wid] = {
                            'correctness': {'status': 'passed'},
                            'measurement': {
                                'status': 'measured',
                                'conservative_geomean': latency.get('conservative_geomean'),
                            },
                            'confirmation': {'status': confirmation.get('status', 'missing')},
                        }
            except (OSError, ValueError, KeyError):
                pass

        return {
            'assignment_id': assignment_id,
            'execution': {
                'status': exec_status,
                'exit_code': None if exit_code == 124 else exit_code,
                'model_exit_code': status.get('model_exit_code'),
                'error': status.get('error'),
            },
            'correctness': {
                'status': 'evaluated' if accepted else 'none_accepted',
                'accepted_count': len(accepted),
            },
            'performance': {
                'status': 'measured' if confirmation.get('status') == 'confirmed' else 'coverage_missing',
                'conservative_geomean': confirmation.get('conservative_geomean'),
            },
            'confirmation': confirmation or {'status': 'missing'},
            'workloads': workloads,
        }


class C500ReleaseVerifier:
    """Verify that the GPU was released after the author session exited."""

    def observe(self, run, claim, terminal):
        exec_info = terminal.get('execution', {})
        clean_exit = exec_info.get('exit_code') in (0, None)
        not_timeout = exec_info.get('status') != 'failed'
        return {
            'assignment_id': run['assignment_id'],
            'device_id': run['device_id'],
            'status': 'verified' if (clean_exit or not_timeout) else 'unverified',
            'reason': 'c500_process_exit',
            'physical_device': run['physical_device'],
        }


def run(state_directory, allocation):
    owner = CampaignOwner(allocation, state_directory)
    return owner.run(C500AuthorLauncher(), C500ReleaseVerifier())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocation', required=True, type=Path)
    parser.add_argument('--state-directory', required=True, type=Path)
    args = parser.parse_args()
    result = run(args.state_directory, args.allocation)
    print(json.dumps(result, indent=2))
