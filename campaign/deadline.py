"""Independent deadline guardian; no provider, device, oracle or result access."""
import os
from pathlib import Path
import select
import signal
import sys
import time


def main():
    pid, stop, expected_ticks = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    cutoff = time.monotonic() + max(0, stop - time.time())
    # Pipe closure also terminates the owned group after controller loss.
    while time.monotonic() < cutoff:
        readable, _, _ = select.select([sys.stdin.buffer], [], [], max(0, cutoff - time.monotonic()))
        if readable and not os.read(0, 1):
            break
    if sys.platform == 'linux':
        try:
            observed = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
            if observed != expected_ticks:
                return
        except FileNotFoundError:
            # A reaped leader can leave a group. Its number cannot be reused while
            # that group still contains descendants.
            pass
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


if __name__ == '__main__':
    main()
