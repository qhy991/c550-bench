"""Bounded Linux process custody and narrow bubblewrap mounts."""
from __future__ import annotations

from dataclasses import dataclass
import os
import ctypes
from pathlib import Path
import signal
import subprocess
import time

from .binding import read_object
from .owner import publish_new


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_ticks: str

    @classmethod
    def observe(cls, pid: int):
        text = Path(f'/proc/{pid}/stat').read_text()
        return cls(pid, text.rsplit(')', 1)[1].split()[19])

    def alive(self):
        try:
            return self == self.observe(self.pid)
        except (FileNotFoundError, ProcessLookupError):
            return False


def parent_death_signal():
    parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'cannot bind child to owner lifetime')
    if os.getppid() != parent or parent == 1:
        os.kill(os.getpid(), signal.SIGKILL)


def terminate_group(process: subprocess.Popen) -> None:
    for sig, seconds in ((signal.SIGTERM, 2), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            continue
    process.wait(timeout=5)


def clean_environment() -> dict:
    return {'PATH': '/opt/conda/bin:/opt/maca/bin:/usr/bin:/bin',
            'LANG': 'C.UTF-8', 'HOME': '/home/author', 'PYTHONDONTWRITEBYTECODE': '1',
            'TORCH_EXTENSIONS_DIR': '/tmp/torch-extensions', 'TRITON_CACHE_DIR': '/tmp/triton',
            'TMPDIR': '/tmp'}


def runtime_environment(root: Path) -> dict:
    host = read_object(root / '.deps/cake-ir/runtime/hosts/xcore1002.json')
    if host.get('target') != 'xcore1002' or host['host_environment']['kind'] != 'maca':
        raise ValueError('selected Compiler has no exact MACA host capture')
    environment = clean_environment()
    environment.update(host['host_environment']['runtime']['build_environment'])
    return environment


def sandbox(root: Path, source: Path, *, writable: list[Path], extra_readonly=(),
            gpu: bool = False, protocol: dict) -> list[str]:
    """Expose no sibling assignment, owner state, host home, or host process namespace."""
    deployment = protocol['deployment']
    command = [deployment['bubblewrap'], '--die-with-parent', '--new-session',
               '--unshare-pid', '--unshare-ipc', '--unshare-uts', '--cap-drop', 'ALL',
               '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp',
               '--tmpfs', '/home', '--dir', '/home/author']
    for name in ('/usr', '/bin', '/lib', '/lib64', '/opt'):
        path = Path(name)
        if path.exists():
            command += ['--ro-bind', name, name]
    for path in (Path('/etc/ld.so.cache'), Path('/etc/resolv.conf'), Path('/etc/hosts'), Path('/etc/ssl')):
        if path.exists():
            command += ['--ro-bind', str(path), str(path)]
    for path in dict.fromkeys((source, root, *map(Path, extra_readonly))):
        command += ['--ro-bind', str(path), str(path)]
    for path in writable:
        command += ['--bind', str(path), str(path)]
    if gpu:
        command += ['--unshare-net']
        device = read_object(root / 'campaign/binding.json')['physical_device']
        route = deployment['devices'][str(device)]
        for path in (deployment['control_device_node'], route['render_node']):
            command += ['--dev-bind', path, path]
        uid = os.geteuid()
        for path in (f'/tmp/open-cake-ir-maca-{uid}.lock',
                     f'/tmp/open-cake-ir-maca-{uid}-device-{device}.lock'):
            if not Path(path).is_file() or Path(path).is_symlink():
                raise ValueError('canonical MACA locks must be mounted from the host before intake')
            command += ['--bind', path, path]
    else:
        for name in ('hmz_root', 'hmz_python_root', 'provider_native'):
            path = deployment[name]
            command += ['--ro-bind', path, path]
    return [*command, '--chdir', str(root)]


def run_bounded(command: list[str], *, cwd: Path, env: dict, stop: float,
                log: Path, process_record: Path, service=None) -> int:
    if time.time() >= stop:
        raise TimeoutError('process opportunity consumed after deadline')
    with log.open('x') as output:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True,
                                   preexec_fn=parent_death_signal)
        try:
            identity = ProcessIdentity.observe(process.pid)
            publish_new(process_record, {'pid': identity.pid, 'start_ticks': identity.start_ticks})
            while process.poll() is None:
                if time.time() >= stop:
                    raise TimeoutError('process deadline expired')
                if service is not None:
                    service()
                time.sleep(0.1)
            return process.returncode
        finally:
            # A completed leader can leave children. The private PID namespace and
            # this process group are owned by this invocation only.
            terminate_group(process)
