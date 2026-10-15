"""Trusted caller for a persistent, untrusted complete-task process."""
from __future__ import annotations

import os
from pathlib import Path
import selectors
import struct
import subprocess
import time

from .owner import publish_new
from .processes import DeadlineGuard, ProcessIdentity, parent_death_signal, terminate_group
from .wire import HEADER_LIMIT, decode, encode


class CandidateSession:
    def __init__(self, command: list[str], *, environment: dict, directory: Path,
                 stop: float, hard_stop: float, maximum_output_bytes: int, device: str):
        self.stop, self.hard_stop = stop, hard_stop
        self.maximum = maximum_output_bytes + HEADER_LIMIT + 4
        self.device = device
        self.failure = None
        if time.time() >= stop or stop > hard_stop:
            raise TimeoutError('candidate has no execution time remaining')
        self.log = (directory / 'worker.log').open('x')
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, env=environment, close_fds=True,
                                        start_new_session=True, preexec_fn=parent_death_signal)
        self.guard = DeadlineGuard(self.process, stop)
        identity = ProcessIdentity.observe(self.process.pid)
        publish_new(directory / 'worker-local-process.json', {
            'pid': identity.pid, 'start_ticks': identity.start_ticks,
            'scope': 'assay PID namespace; host release observes the enclosing namespace leader'})
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        try:
            terminate_group(self.process, stop=self.hard_stop)
        finally:
            self.guard.close(self.hard_stop)
            self.process.stdin.close()
            self.process.stdout.close()
            self.log.close()

    def __call__(self, *inputs):
        try:
            return self.exchange(inputs)
        except Exception as error:
            self.failure = f'{type(error).__name__}: {error}'
            raise

    def exchange(self, inputs):
        outgoing = memoryview(encode(list(inputs)))
        received = bytearray()
        expected = None
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdin, selectors.EVENT_WRITE)
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while expected is None or len(received) < expected + 8 or outgoing:
                remaining = self.stop - time.time()
                if remaining <= 0:
                    raise TimeoutError('candidate exceeded its original deadline')
                for key, _ in selector.select(remaining):
                    if key.fileobj is self.process.stdin:
                        written = os.write(key.fd, outgoing[:65536])
                        outgoing = outgoing[written:]
                        if not outgoing:
                            selector.unregister(self.process.stdin)
                    else:
                        block = os.read(key.fd, min(65536, self.maximum + 9 - len(received)))
                        if not block:
                            raise RuntimeError('candidate exited without a complete value; no retry')
                        received.extend(block)
                        if len(received) >= 8 and expected is None:
                            expected, = struct.unpack('!Q', received[:8])
                            if expected > self.maximum:
                                raise ValueError('candidate response exceeds output contract')
                        if expected is not None and len(received) > expected + 8:
                            raise ValueError('candidate sent trailing data')
        if time.time() >= self.stop:
            raise TimeoutError('candidate response arrived after deadline')
        return decode(bytes(received[8:]), device=self.device)
