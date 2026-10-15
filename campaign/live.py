"""Source-bound live admission, concrete launcher and canonical release observation."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

from .binding import clean_checkout, read_object, reconcile_allocation
from .owner import CampaignOwner, publish_new
from .processes import ProcessIdentity, clean_environment, terminate_group, parent_death_signal, DeadlineGuard, admit_private_paths

SOURCE = Path(__file__).resolve().parents[1]


def admit_live(allocation: Path, state: Path) -> dict:
    protocol = read_object(SOURCE / 'campaign/protocol.json')
    if (protocol.get('live_execution') is not True or protocol.get('status') != 'qualified'
            or not protocol.get('qualification_evidence')
            or any(protocol.get(name, {}).get('status') != 'qualified'
                   for name in ('timing', 'profiling', 'release', 'isolation'))):
        raise ValueError('source-bound C550 whole-call timing, profile, isolation and release remain unqualified')
    if sys.platform != 'linux' or not os.statvfs(SOURCE).f_flag & os.ST_RDONLY:
        raise PermissionError('live Bench source requires a Linux read-only deployment')
    document = read_object(allocation)
    admit_private_paths([state, *(run['root'] for run in document.get('runs', []))])
    reconcile_allocation(document)
    if document.get('host') != 'c550-2':
        raise ValueError('this deployment admits only the separately qualified c550-2 host')
    tasks = {row['id'] for row in read_object(SOURCE / 'suite.json')['tasks']}
    if {run['task'] for run in document.get('runs', [])} != tasks:
        raise ValueError('live allocation must retain all twelve original tasks')
    for run in document.get('runs', []):
        clean_checkout(SOURCE, run['bench_commit'])
        root = clean_checkout(Path(run['root']), run['prepared_commit'])
        clean_checkout(root / '.deps/cake-ir', run['compiler_commit'])
        clean_checkout(root / '.deps/sol-execbench', read_object(root / 'sources.lock.json')['sol_execbench']['revision'])
        if (run['measurement_protocol'] != 'campaign/protocol.json'
                or read_object(root / run['measurement_protocol']) != protocol
                or run['author']['model'] != 'glm-5.3:high'
                or run['author']['knowledge'] != 'none'
                or run['author']['reference_access'] != 'known_kernel_reproduction'
                or run['author']['scaffold'] != 'campaign/TASK.md'
                or run['correctness_rounds'] != 10 or len(run['workload_ids']) != 16):
            raise ValueError('prepared assignment differs from the qualified whole-task author protocol')
        route = protocol['deployment']['devices'].get(str(run['physical_device']))
        if not route:
            raise ValueError('physical C550 device lacks source-bound namespace mapping')
    return protocol


class SupervisorHandle:
    def __init__(self, process, stop, hard_stop, log):
        self.process, self.stop, self.hard_stop, self.log = process, stop, hard_stop, log
        self.guard = DeadlineGuard(process, stop)

    def poll(self):
        code = self.process.poll()
        if code is None and time.time() >= self.stop:
            terminate_group(self.process, stop=self.hard_stop)
            code = self.process.returncode
        if code is not None:
            self.guard.close(self.hard_stop)
            self.log.close()
        return code


class LiveLauncher:
    def __init__(self, state: Path, protocol: dict):
        self.state, self.protocol = state, protocol

    def start(self, run: dict, claim: dict):
        command = [sys.executable, '-I', str(SOURCE / 'campaign/entry.py'),
                   'supervise', '--root', run['root'], '--state', str(self.state)]
        log_path = self.state / 'logs' / f"{run['assignment_id']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open('x')
        try:
            process = subprocess.Popen(command, env=clean_environment(), cwd=SOURCE,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                       preexec_fn=parent_death_signal)
            identity = ProcessIdentity.observe(process.pid)
            publish_new(self.state / 'supervisors' / f"{run['assignment_id']}.json",
                        {'pid': identity.pid, 'start_ticks': identity.start_ticks})
            handle = SupervisorHandle(process, claim['execution_stop_at_epoch'], claim['stop_at_epoch'], log)
        except BaseException:
            if 'process' in locals():
                terminate_group(process, stop=claim['stop_at_epoch'])
            log.close()
            raise
        return handle

    def terminal(self, run: dict, handle, exit_code: int):
        if exit_code != 0:
            raise RuntimeError(f'supervisor exited {exit_code}; claimed opportunity is consumed')
        return read_object(self.state / 'assignments' / run['assignment_id'] / 'terminal.json')


class CanonicalRelease:
    def __init__(self, state: Path):
        self.state = state

    def observe(self, run: dict, claim: dict, terminal: dict):
        folder = self.state / 'assignments' / run['assignment_id']
        supervisor = self.state / 'supervisors' / f"{run['assignment_id']}.json"
        if not supervisor.is_file():
            return self.unverified(run, 'supervisor launch custody is unknown')
        records = [supervisor, *folder.glob('**/process.json'), *folder.glob('author-process.json')]
        for path in records:
            identity = read_object(path)
            if ProcessIdentity(**identity).alive():
                return self.unverified(run, 'an owned process is still alive')
        output = self.state / 'release-observations' / f"{run['assignment_id']}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-I', str(SOURCE / 'campaign/entry.py'), 'release',
                   '--root', run['root'], '--state', str(self.state), '--request', str(output)]
        remaining = claim['stop_at_epoch'] - time.time()
        if remaining <= 0:
            return self.unverified(run, 'hard deadline elapsed before canonical release observation')
        code = subprocess.run(command, env=clean_environment(), cwd=SOURCE,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              timeout=min(15, remaining)).returncode
        if code != 0 or not output.is_file():
            return self.unverified(run, 'canonical lock release observation failed')
        result = read_object(output)
        return {'assignment_id': run['assignment_id'], 'device_id': run['device_id'],
                'status': result['status'], 'processes_observed_exited': len(records),
                'canonical_observation': str(output), 'observation': result}

    @staticmethod
    def unverified(run, reason):
        return {'assignment_id': run['assignment_id'], 'device_id': run['device_id'],
                'status': 'unverified', 'reason': reason}


def observe_release(root: Path, output: Path):
    binding = read_object(root / 'campaign/binding.json')
    clean_checkout(root / '.deps/cake-ir', binding['compiler_commit'])
    sys.path.insert(0, str(root / '.deps/cake-ir/src'))
    from open_cake_ir.evaluation.local_broker import _acquire, _lock_path, _device_lock_path
    paths = (_lock_path('maca'), _device_lock_path('maca', binding['physical_device']))
    descriptors, observations = [], []
    try:
        for index, path in enumerate(paths):
            if not path.is_file() or path.is_symlink():
                raise ValueError('canonical MACA lock is unavailable')
            before = path.stat()
            fd = _acquire(path, shared=index == 0)
            descriptors.append(fd)
            held = os.fstat(fd)
            if (before.st_dev, before.st_ino) != (held.st_dev, held.st_ino):
                raise ValueError('canonical lock identity changed during release observation')
            observations.append({'path': str(path), 'device': held.st_dev, 'inode': held.st_ino})
        result = {'status': 'verified', 'basis': 'owned_process_namespaces_exited_and_canonical_locks_acquired',
                  'physical_device': binding['physical_device'], 'locks': observations}
    except (OSError, ValueError) as error:
        result = {'status': 'unverified', 'reason': f'{type(error).__name__}: {error}'}
    finally:
        for fd in reversed(descriptors):
            os.close(fd)
    publish_new(output, result)


def serve(allocation: Path, state: Path):
    allocation = allocation.resolve(strict=True)
    state = state.absolute()
    if state != state.resolve() or any((parent / '.git').exists() for parent in [state, *state.parents]):
        raise ValueError('owner state must be canonical and outside checkouts')
    protocol = admit_live(allocation, state)
    owner = CampaignOwner(allocation, state)
    return owner.run(LiveLauncher(state, protocol), CanonicalRelease(state))


def reconcile(allocation: Path, state: Path):
    owner = CampaignOwner(allocation, state)
    return owner.reconcile(CanonicalRelease(state))
