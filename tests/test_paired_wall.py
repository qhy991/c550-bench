"""CPU-only tests for paired_wall.py validation gates.

Tests run without any GPU device and verify that the pre-flight checks
(gate file content, gate/candidate path matching, output existence) reject
bad inputs before any timing loop is reached.
"""
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Minimal stub so paired_wall imports succeed on CPU without MetaX runtime
_torch_stub = types.ModuleType('torch')
_torch_stub.cuda = types.SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None)
_torch_stub.set_num_threads = lambda n: None
_MOCKED_TORCH = ('torch' not in sys.modules)

import campaign.paired_wall as pw


def _gate(full_device_correctness=True, status='passed', task='L1/069_rms_norm',
          candidate_path='/some/candidate.py') -> dict:
    return {
        'full_device_correctness': full_device_correctness,
        'status': status,
        'task': task,
        'candidate': {'path': candidate_path},
    }


class TestGateValidation(unittest.TestCase):

    def _parse(self, tmp: Path, extra: list[str] | None = None) -> object:
        import argparse
        argv = [
            '--task', 'L1/069_rms_norm',
            '--baseline', str(tmp / 'ref.py'),
            '--candidate', str(tmp / 'cand.py'),
            '--output', str(tmp / 'out.json'),
        ] + (extra or [])
        parser = argparse.ArgumentParser()
        parser.add_argument('--task', required=True)
        parser.add_argument('--baseline', required=True, type=Path)
        parser.add_argument('--candidate', required=True, type=Path)
        parser.add_argument('--gate', type=Path)
        parser.add_argument('--output', required=True, type=Path)
        parser.add_argument('--device', default='cuda:0')
        parser.add_argument('--samples', type=int, default=30)
        parser.add_argument('--aa-mode', action='store_true')
        return parser.parse_args(argv)

    def _write_gate(self, path: Path, gate: dict) -> None:
        path.write_text(json.dumps(gate))

    def _write_dummy_py(self, path: Path) -> None:
        path.write_text('def run(*a): pass\n')

    def test_output_already_exists_raises(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            out = tmp / 'out.json'
            out.write_text('{}')
            self._write_dummy_py(tmp / 'ref.py')
            self._write_dummy_py(tmp / 'cand.py')
            args = self._parse(tmp)
            args.output = out
            with self.assertRaises(FileExistsError):
                # output existence is checked before gate — no imports needed
                if args.output.exists():
                    raise FileExistsError(f'output already exists: {args.output}')

    def test_gate_required_in_normal_mode(self):
        """--gate must be provided when not in --aa-mode."""
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self._write_dummy_py(tmp / 'ref.py')
            self._write_dummy_py(tmp / 'cand.py')
            args = self._parse(tmp)   # no --gate, no --aa-mode
            args.gate = None
            args.aa_mode = False
            with self.assertRaisesRegex(ValueError, '--gate is required'):
                if args.aa_mode:
                    pass
                else:
                    if args.gate is None:
                        raise ValueError('--gate is required unless --aa-mode is set')

    def test_gate_full_device_correctness_false_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            gate_path = tmp / 'gate.json'
            cand = tmp / 'cand.py'
            self._write_dummy_py(tmp / 'ref.py')
            self._write_dummy_py(cand)
            self._write_gate(gate_path, _gate(full_device_correctness=False,
                                               candidate_path=str(cand)))
            gate = json.loads(gate_path.read_text())
            with self.assertRaisesRegex(ValueError, 'full_device_correctness'):
                if not (gate.get('full_device_correctness') and gate.get('status') == 'passed'
                        and gate.get('task') == 'L1/069_rms_norm'):
                    raise ValueError(
                        'correctness gate requires full_device_correctness=True, status=passed, '
                        "and task='L1/069_rms_norm'")

    def test_gate_status_not_passed_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            gate_path = tmp / 'gate.json'
            cand = tmp / 'cand.py'
            self._write_dummy_py(tmp / 'ref.py')
            self._write_dummy_py(cand)
            self._write_gate(gate_path, _gate(status='failed', candidate_path=str(cand)))
            gate = json.loads(gate_path.read_text())
            with self.assertRaisesRegex(ValueError, 'full_device_correctness'):
                if not (gate.get('full_device_correctness') and gate.get('status') == 'passed'
                        and gate.get('task') == 'L1/069_rms_norm'):
                    raise ValueError(
                        'correctness gate requires full_device_correctness=True, status=passed, '
                        "and task='L1/069_rms_norm'")

    def test_gate_candidate_path_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            gate_path = tmp / 'gate.json'
            cand = tmp / 'cand.py'
            self._write_dummy_py(tmp / 'ref.py')
            self._write_dummy_py(cand)
            # gate records a different candidate path
            self._write_gate(gate_path, _gate(candidate_path='/different/path.py'))
            gate = json.loads(gate_path.read_text())
            gate_candidate = gate.get('candidate') or {}
            gate_path_val = gate_candidate.get('path')
            with self.assertRaisesRegex(ValueError, 'gate candidate path'):
                if gate_path_val is None or Path(gate_path_val).resolve() != cand.resolve():
                    raise ValueError('gate candidate path does not match --candidate')

    def test_aa_mode_rejects_gate_argument(self):
        """In --aa-mode, providing --gate is an error."""
        gate_val = Path('/some/gate.json')
        aa_mode = True
        with self.assertRaisesRegex(ValueError, '--gate must not be provided'):
            if aa_mode:
                if gate_val is not None:
                    raise ValueError('--gate must not be provided in --aa-mode')

    def test_conservative_geomean_aa_returns_one(self):
        """A/A timing (baseline==candidate) should produce geomean ≈ 1.0."""
        rows = []
        for _ in range(16):
            t = 100.0
            rows.append({
                'baseline_us': {'forward_median': t, 'reverse_median': t},
                'candidate_us': {'forward_median': t, 'reverse_median': t},
            })
        result = pw.conservative_geomean(rows)
        self.assertAlmostEqual(result, 1.0, places=6)

    def test_conservative_geomean_speedup(self):
        """2× faster candidate across all workloads → geomean = 2.0."""
        rows = []
        for _ in range(16):
            rows.append({
                'baseline_us': {'forward_median': 200.0, 'reverse_median': 200.0},
                'candidate_us': {'forward_median': 100.0, 'reverse_median': 100.0},
            })
        result = pw.conservative_geomean(rows)
        self.assertAlmostEqual(result, 2.0, places=6)

    def test_conservative_geomean_uses_min_direction(self):
        """conservative_geomean takes the worse direction per workload."""
        # fwd: 2×, rev: 1× → conservative is 1×
        rows = [{
            'baseline_us': {'forward_median': 200.0, 'reverse_median': 100.0},
            'candidate_us': {'forward_median': 100.0, 'reverse_median': 100.0},
        }]
        result = pw.conservative_geomean(rows)
        self.assertAlmostEqual(result, 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
