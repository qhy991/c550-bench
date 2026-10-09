"""Suite extension preserves prior receipts and refuses changed source data."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('c550_prepare', ROOT / 'scripts/prepare.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class SuiteExtension(unittest.TestCase):
    def setUp(self):
        self.suite = json.loads((ROOT / 'suite.json').read_text())
        self.lock = json.loads((ROOT / 'sources.lock.json').read_text())
        self.rows = []
        for task in self.suite['tasks']:
            workloads = [{'uuid': task['smoke_workload_uuid'], 'axes': {'n': 2}, 'inputs': {'x': {'type': 'random'}}}]
            workloads += [{'uuid': f"fixture-{task['id']}-{i}", 'axes': {'n': 2},
                           'inputs': {'x': {'type': 'random'}}} for i in range(15)]
            self.rows.append({'name': task['id'].split('/', 1)[1], 'description': 'Synthetic preparation control',
                'axes': json.dumps({'n': {'type': 'var'}}), 'inputs': json.dumps({'x': {'shape': ['n'], 'dtype': 'float32'}}),
                'outputs': json.dumps({'y': {'shape': ['n'], 'dtype': 'float32'}}),
                'reference': 'def run(x):\n    return x\n', 'workloads': json.dumps(workloads)})

    def run_prepare(self, root):
        parquet = ModuleType('pyarrow.parquet')
        parquet.read_table = lambda path: SimpleNamespace(to_pylist=lambda: self.rows)
        arrow = ModuleType('pyarrow')
        arrow.parquet = parquet
        with patch.object(prepare, 'ROOT', root), patch.object(prepare, 'ensure_source'), \
                patch.object(prepare, 'download'), patch.dict(sys.modules, {'pyarrow': arrow, 'pyarrow.parquet': parquet}), \
                patch.object(sys, 'argv', ['prepare.py']), contextlib.redirect_stdout(io.StringIO()):
            prepare.main()

    def make_root(self, directory, suite):
        root = Path(directory)
        (root / 'suite.json').write_text(json.dumps(suite))
        (root / 'sources.lock.json').write_text(json.dumps(self.lock))
        return root

    def test_catalog_has_the_two_matching_gemm_tasks_and_c550_identity(self):
        self.assertEqual(self.suite['version'], 2)
        self.assertEqual(self.suite['target_arch'], 'metax_c550')
        self.assertEqual(self.suite['target_device_name'], 'MetaX C550')
        self.assertEqual(self.suite['seed'], 200)
        self.assertEqual(self.suite['correctness_rounds'], 10)
        self.assertEqual(len(self.suite['tasks']), 12)
        self.assertEqual(len({task['id'] for task in self.suite['tasks']}), 12)
        self.assertEqual([task['id'] for task in self.suite['tasks'][-2:]], [
            'L1/003_lm_head_projection_with_logit_slicing', 'L1/077_whisper_decoder_output_projection'])
        self.assertTrue(all(task['family'] == 'gemm' for task in self.suite['tasks'][-2:]))

    def test_v2_materialization_keeps_v1_receipt_and_original_files(self):
        with tempfile.TemporaryDirectory() as directory:
            original = {**self.suite, 'version': 1, 'tasks': self.suite['tasks'][:10]}
            root = self.make_root(directory, original)
            self.run_prepare(root)
            receipt = root / '.data/materialization.json'
            before = receipt.read_bytes()
            old_reference = root / '.data/benchmark' / original['tasks'][0]['id'] / 'reference.py'
            old_bytes = old_reference.read_bytes()
            (root / 'suite.json').write_text(json.dumps(self.suite))
            self.run_prepare(root)
            self.assertEqual(receipt.read_bytes(), before)
            self.assertEqual(old_reference.read_bytes(), old_bytes)
            current = json.loads((root / '.data/materialization-suite-v2.json').read_text())
            self.assertEqual(len(current['tasks']), 12)
            self.assertEqual(sum(row['workloads'] for row in current['tasks']), 192)
            self.run_prepare(root)  # Identical preparation converges without replacing old records.
            self.assertEqual(receipt.read_bytes(), before)

    def test_changed_reference_is_refused_during_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            original = {**self.suite, 'version': 1, 'tasks': self.suite['tasks'][:10]}
            root = self.make_root(directory, original)
            self.run_prepare(root)
            reference = root / '.data/benchmark' / original['tasks'][0]['id'] / 'reference.py'
            reference.write_text('changed original reference')
            (root / 'suite.json').write_text(json.dumps(self.suite))
            with self.assertRaisesRegex(RuntimeError, 'Refusing to replace changed data'):
                self.run_prepare(root)
            self.assertEqual(reference.read_text(), 'changed original reference')
            self.assertFalse((root / '.data/materialization-suite-v2.json').exists())

    def test_duplicate_workloads_are_refused_before_publication(self):
        rows = json.loads(self.rows[0]['workloads'])
        rows[-1] = rows[0]
        self.rows[0]['workloads'] = json.dumps(rows)
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(directory, self.suite)
            with self.assertRaisesRegex(RuntimeError, '16 distinct original workloads'):
                self.run_prepare(root)
            self.assertFalse((root / '.data/materialization-suite-v2.json').exists())


if __name__ == '__main__':
    unittest.main()
