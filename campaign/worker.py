"""Untrusted complete-call worker. Its only authority is returned value bytes."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
import sys

# This standalone file and wire.py are the only Bench files in the worker jail.
sys.path.insert(0, str(Path(__file__).parent))
from wire import decode, encode


def read_exact(stream, size):
    chunks = bytearray()
    while len(chunks) < size:
        part = stream.read(size - len(chunks))
        if not part:
            raise EOFError('call input closed')
        chunks.extend(part)
    return bytes(chunks)


def main():
    bundle, device, compiler = sys.argv[1:]
    sys.path[:0] = [bundle, str(Path(compiler) / 'src')]
    import torch
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location('submitted_candidate', Path(bundle) / 'candidate.py')
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    while True:
        prefix = sys.stdin.buffer.read(8)
        if not prefix:
            return
        if len(prefix) != 8:
            raise EOFError('truncated frame')
        size, = struct.unpack('!Q', prefix)
        inputs = decode(read_exact(sys.stdin.buffer, size), device=device, preserve_strides=True)
        output = candidate.run(*inputs)
        if device != 'cpu':
            torch.cuda.synchronize(0)
        sys.stdout.buffer.write(encode(output))
        sys.stdout.buffer.flush()
        del inputs, output


if __name__ == '__main__':
    main()
