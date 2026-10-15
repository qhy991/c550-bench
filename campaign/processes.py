"""Bounded Linux process custody and narrow bubblewrap mounts."""
from __future__ import annotations

from dataclasses import dataclass
import os
import ctypes
from pathlib import Path
import signal
import subprocess
import sys
import time

from .binding import read_object
from .owner import publish_new

RUNTIME_READONLY = ('/usr', '/bin', '/lib', '/lib64', '/opt/conda', '/opt/maca', '/etc/ld.so.cache')


def runtime_paths(protocol: dict, *, author=False):
    paths = list(RUNTIME_READONLY)
    if author:
        paths.extend(('/etc/resolv.conf', '/etc/hosts', '/etc/ssl'))
        paths.extend(protocol['deployment'][name] for name in ('hmz_root', 'hmz_python_root', 'provider_native'))
    return tuple(dict.fromkeys(paths))


def admit_private_paths(paths, protocol: dict, source: Path) -> None:
    """Admit layouts against every dependency exposed to an untrusted role."""
    private = [Path(path).resolve() for path in paths]
    for index, path in enumerate(private):
        if any(path.is_relative_to(other) or other.is_relative_to(path) for other in private[:index]):
            raise ValueError('private campaign paths overlap: ' + str(path))
    for name in (*runtime_paths(protocol, author=True), source):
        runtime = Path(name).resolve()
        for path in private:
            if path.is_relative_to(runtime):
                raise ValueError('private campaign path overlaps untrusted runtime mount: ' + str(path))


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


def terminate_group(process: subprocess.Popen, *, stop: float) -> None:
    """Kill immediately; the caller's existing deadline includes reaping."""
    process.poll()
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=max(0, stop - time.time()))


class DeadlineGuard:
    def __init__(self, process, stop):
        try:
            identity = ProcessIdentity.observe(process.pid)
            self.guard = subprocess.Popen(
                [sys.executable, '-I', str(Path(__file__).with_name('deadline.py')),
                 str(process.pid), str(stop), identity.start_ticks], stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True, close_fds=True)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise

    def close(self, stop):
        self.guard.stdin.close()
        self.guard.wait(timeout=max(0, stop - time.time()))


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
            gpu: bool = False, protocol: dict, author: bool = False) -> list[str]:
    """Expose no sibling assignment, owner state, host home, or host process namespace."""
    deployment = protocol['deployment']
    command = [deployment['bubblewrap'], '--die-with-parent', '--new-session',
               '--unshare-pid', '--unshare-ipc', '--unshare-uts', '--cap-drop', 'ALL',
               '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp',
               '--tmpfs', '/home', '--dir', '/home/author']
    for name in runtime_paths(protocol, author=author):
        path = Path(name)
        if path.exists():
            command += ['--ro-bind', name, name]
    if author:
        roots = (source, root)
    else:
        binding = read_object(root / 'campaign/binding.json')
        roots = (source, root / 'suite.json', root / 'sources.lock.json',
                 root / 'campaign/binding.json', root / 'campaign/protocol.json',
                 root / '.deps/cake-ir', root / '.deps/sol-execbench',
                 root / '.data/benchmark' / binding['task'], root / binding['baseline'])
    for path in dict.fromkeys((*roots, *map(Path, extra_readonly))):
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
    return [*command, '--chdir', str(root)]


def candidate_sandbox(root: Path, source: Path, bundle: Path, protocol: dict) -> list[str]:
    """The worker sees only its immutable bundle and explicit runtime dependencies."""
    deployment = protocol['deployment']
    command = [deployment['bubblewrap'], '--die-with-parent', '--new-session',
               '--unshare-pid', '--unshare-ipc', '--unshare-uts', '--unshare-net',
               '--cap-drop', 'ALL', '--proc', '/proc', '--dev', '/dev',
               '--tmpfs', '/tmp', '--tmpfs', '/home', '--dir', '/home/author']
    for name in runtime_paths(protocol):
        if Path(name).exists():
            command += ['--ro-bind', name, name]
    for name in ('worker.py', 'wire.py'):
        command += ['--ro-bind', str(source / 'campaign' / name), '/worker/' + name]
    command += ['--ro-bind', str(bundle), '/bundle',
                '--ro-bind', str(root / '.deps/cake-ir'), '/compiler']
    device = read_object(root / 'campaign/binding.json')['physical_device']
    route = deployment['devices'][str(device)]
    for path in (deployment['control_device_node'], route['render_node']):
        command += ['--dev-bind', path, path]
    return [*command, '--chdir', '/bundle', deployment['assay_python'], '-I',
            '/worker/worker.py', '/bundle', 'cuda:0', '/compiler']


def run_bounded(command: list[str], *, cwd: Path, env: dict, stop: float,
                log: Path, process_record: Path, service=None, hard_stop: float | None = None) -> int:
    hard_stop = stop if hard_stop is None else hard_stop
    if stop > hard_stop:
        raise ValueError('execution cannot exceed its hard deadline')
    if time.time() >= stop:
        raise TimeoutError('process opportunity consumed after deadline')
    with log.open('x') as output:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True,
                                   preexec_fn=parent_death_signal)
        guard = DeadlineGuard(process, stop)
        try:
            identity = ProcessIdentity.observe(process.pid)
            publish_new(process_record, {'pid': identity.pid, 'start_ticks': identity.start_ticks})
            while process.poll() is None:
                if time.time() >= stop:
                    raise TimeoutError('process deadline expired')
                if service is not None:
                    service()
                time.sleep(min(0.1, max(0, stop - time.time())))
            if time.time() >= stop:
                raise TimeoutError('process deadline expired')
            return process.returncode
        finally:
            # A completed leader can leave children. The private PID namespace and
            # this process group are owned by this invocation only.
            try:
                terminate_group(process, stop=hard_stop)
            finally:
                guard.close(hard_stop)
