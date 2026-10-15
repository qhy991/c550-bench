"""Bounded tensor/value transport. No pickle, imports, paths or shared storage."""
from __future__ import annotations

import json
import math
import struct

HEADER_LIMIT = 1024 * 1024
DTYPES = ('bool', 'uint8', 'int8', 'int16', 'int32', 'int64',
          'float16', 'bfloat16', 'float32', 'float64')


def encode(value) -> bytes:
    import torch
    buffers = []

    def visit(item):
        if isinstance(item, torch.Tensor):
            name = str(item.dtype).removeprefix('torch.')
            if name not in DTYPES or item.layout != torch.strided:
                raise ValueError('wire accepts only dense tensors with admitted dtypes')
            data = item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            buffers.append(data)
            return ['tensor', name, list(item.shape), list(item.stride()), len(data)]
        if item is None or type(item) in (bool, int, float, str):
            return ['scalar', item]
        if type(item) in (list, tuple):
            return [type(item).__name__, [visit(child) for child in item]]
        if type(item) is dict and all(type(key) is str for key in item):
            return ['dict', [[key, visit(child)] for key, child in item.items()]]
        raise ValueError('callable ABI contains an unsupported value')

    header = json.dumps(visit(value), allow_nan=False, separators=(',', ':')).encode()
    if len(header) > HEADER_LIMIT:
        raise ValueError('wire metadata exceeds limit')
    payload = struct.pack('!I', len(header)) + header + b''.join(buffers)
    return struct.pack('!Q', len(payload)) + payload


def decode(payload: bytes, *, device: str, preserve_strides: bool = False):
    import torch
    if len(payload) < 4:
        raise ValueError('truncated wire header')
    size, = struct.unpack('!I', payload[:4])
    if size > HEADER_LIMIT or size > len(payload) - 4:
        raise ValueError('wire metadata exceeds limit')
    tree = json.loads(payload[4:4 + size])
    position = 4 + size

    def visit(node, depth=0):
        nonlocal position
        if depth > 32 or not isinstance(node, list) or not node:
            raise ValueError('invalid wire tree')
        kind = node[0]
        if kind == 'tensor' and len(node) == 5:
            _, name, shape, strides, length = node
            if (name not in DTYPES or not isinstance(shape, list) or len(shape) > 32
                    or any(type(x) is not int or x < 0 for x in shape)
                    or not isinstance(strides, list) or len(strides) != len(shape)
                    or any(type(x) is not int or x < 0 for x in strides)
                    or type(length) is not int or length < 0):
                raise ValueError('invalid wire tensor')
            dtype = getattr(torch, name)
            required = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            if length != required or length > len(payload) - position:
                raise ValueError('wire tensor length differs')
            if length:
                value = torch.frombuffer(bytearray(payload[position:position + length]), dtype=dtype).reshape(shape)
            else:
                value = torch.empty(shape, dtype=dtype)
            position += length
            value = value.to(device)
            # Only trusted inputs retain strides. Candidate output metadata cannot
            # cause an allocation larger than its bounded contiguous bytes.
            if preserve_strides:
                result = torch.empty_strided(shape, strides, dtype=dtype, device=device)
                result.copy_(value)
                return result
            return value
        if kind == 'scalar' and len(node) == 2 and (
                node[1] is None or type(node[1]) in (bool, int, float, str)):
            return node[1]
        if kind in ('list', 'tuple') and len(node) == 2 and isinstance(node[1], list):
            values = [visit(child, depth + 1) for child in node[1]]
            return tuple(values) if kind == 'tuple' else values
        if kind == 'dict' and len(node) == 2 and isinstance(node[1], list):
            values = {}
            for pair in node[1]:
                if (not isinstance(pair, list) or len(pair) != 2 or type(pair[0]) is not str
                        or pair[0] in values):
                    raise ValueError('invalid wire dictionary')
                values[pair[0]] = visit(pair[1], depth + 1)
            return values
        raise ValueError('invalid wire value')

    result = visit(tree)
    if position != len(payload):
        raise ValueError('trailing wire bytes')
    return result
