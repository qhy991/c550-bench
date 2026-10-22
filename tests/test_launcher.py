"""CPU-only tests for the bench launcher components (no GPU required)."""
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure c550-bench-clone root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _make_binding(tmp, physical_device=0, task='L1/069_rms_norm',
                  correctness_rounds=10):
    d = {
        'schema_version': 1,
        'kind': 'c550_bench_whole_task',
        'assignment_id': 'test-001-c-r1',
        'frozen_workspace': '/var/tmp/fake/test-001-control',
        'task': task,
        'condition': 'control',
        'replicate': 1,
        'root': str(tmp),
        'host': 'test-node',
        'device_id': 'c500-0',
        'physical_device': physical_device,
        'author_capacity': 1,
        'bench_commit': 'abc123',
        'compiler_commit': 'def456',
        'hmz_adapter_source_commit': 'ghi789',
        'author': {
            'model': 'glm-5.3:high',
            'scaffold': 'campaign/author-scaffold-v5-restricted.md',
            'knowledge': 'v5-event-contract-restricted',
            'reference_access': 'known_kernel_reproduction',
            'wall_time_seconds': 10800,
            'confirmation_seconds': 1080,
        },
        'measurement_protocol': 'campaign/protocol.json',
        'baseline': 'examples/l1_069_torch_baseline.py',
        'workload_ids': ['uuid-' + str(i) for i in range(16)],
        'correctness_rounds': correctness_rounds,
    }
    (tmp / 'campaign').mkdir(parents=True, exist_ok=True)
    (tmp / 'campaign/binding.json').write_text(json.dumps(d))
    return d


def _make_deadline(tmp, wall=10800, reserve=1080):
    now = time.time()
    d = {
        'started_at_epoch': now,
        'search_stop_at_epoch': now + wall - reserve,
        'stop_at_epoch': now + wall,
        'wall_time_seconds': wall,
        'confirmation_seconds': reserve,
        'selected_physical_device': 0,
        'preparation_cost_included': True,
    }
    (tmp / 'campaign').mkdir(parents=True, exist_ok=True)
    (tmp / 'campaign/deadline.json').write_text(json.dumps(d))
    return d


def _make_frozen_plan(tmp, condition='control', bench_commit='abc123',
                      compiler_commit='def456'):
    d = {
        'schema_version': 1,
        'kind': 'rolling_bench_fresh_search',
        'target': 'metax_c500',
        'bindings': {
            'bench': {'commit': bench_commit},
            'control': {'commit': compiler_commit},
            'successor': {'commit': 'successor-' + compiler_commit},
        },
        'task_ids': ['L1/069_rms_norm'],
        'replicates': 1,
        'author': {
            'model': 'glm-5.3:high',
            'scaffold': 'campaign/author-scaffold-v5-restricted.md',
            'knowledge': 'v5-event-contract-restricted',
            'reference_access': 'known_kernel_reproduction',
            'wall_time_seconds': 10800,
            'confirmation_seconds': 1080,
        },
        'measurement_protocol': 'campaign/protocol.json',
        'baseline_files': {'L1/069_rms_norm': 'examples/l1_069_torch_baseline.py'},
        'allocations': [
            {'task': 'L1/069_rms_norm', 'condition': 'control', 'replicate': 1,
             'workspace': '/var/tmp/fake/test-001-control'},
            {'task': 'L1/069_rms_norm', 'condition': 'successor', 'replicate': 1,
             'workspace': '/var/tmp/fake/test-001-successor'},
        ],
        'replicates': 1,
        'total_author_wall_seconds': 21600,
    }
    (tmp / 'campaign/frozen-plan.json').write_text(json.dumps(d))
    return d


# ---------------------------------------------------------------------------
# evaluate.py tests
# ---------------------------------------------------------------------------

class TestEvaluateLaunchChecks(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_campaign(self):
        binding = _make_binding(self.tmpdir)
        _make_deadline(self.tmpdir)
        _make_frozen_plan(self.tmpdir)
        (self.tmpdir / 'campaign/evaluations').mkdir(parents=True, exist_ok=True)
        (self.tmpdir / 'campaign/candidates').mkdir(parents=True, exist_ok=True)
        return binding

    def test_expired_deadline_rejected(self):
        """evaluate.py refuses to start if the search budget is already expired."""
        self._write_campaign()
        # Overwrite deadline with expired one
        now = time.time()
        deadline = {
            'started_at_epoch': now - 3600,
            'search_stop_at_epoch': now - 100,
            'stop_at_epoch': now + 100,
            'wall_time_seconds': 3500,
            'confirmation_seconds': 100,
        }
        (self.tmpdir / 'campaign/deadline.json').write_text(json.dumps(deadline))
        (self.tmpdir / 'campaign/evaluations').mkdir(parents=True, exist_ok=True)

        # Import evaluate.py from the project root
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'evaluate', ROOT / 'campaign/evaluate.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Write a dummy candidate file
        cand = self.tmpdir / 'campaign/candidates/cand_a.py'
        cand.write_text('def run(*a): pass\n')

        # Patch os.chdir and sys.argv, then call main
        with patch('sys.argv', ['evaluate.py', '--candidate',
                                'campaign/candidates/cand_a.py', '--id', 'a001']):
            orig_cwd = os.getcwd()
            try:
                os.chdir(self.tmpdir)
                # Monkeypatch ROOT inside the module to point to tmpdir
                orig_root = mod.ROOT
                mod.ROOT = self.tmpdir
                try:
                    ret = mod.main()
                finally:
                    mod.ROOT = orig_root
            finally:
                os.chdir(orig_cwd)
        self.assertEqual(ret, 1)
        outcome = json.loads((self.tmpdir / 'campaign/evaluations/a001/outcome.json').read_text())
        self.assertEqual(outcome['status'], 'not_qualified')
        self.assertIn('expired', outcome['reason'])

    def test_duplicate_id_rejected(self):
        """evaluate.py refuses to create an evaluation if the id folder already exists."""
        self._write_campaign()
        existing = self.tmpdir / 'campaign/evaluations/dup001'
        existing.mkdir(parents=True)

        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'evaluate2', ROOT / 'campaign/evaluate.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        cand = self.tmpdir / 'campaign/candidates/cand_dup.py'
        cand.write_text('def run(*a): pass\n')

        with patch('sys.argv', ['evaluate.py', '--candidate',
                                'campaign/candidates/cand_dup.py', '--id', 'dup001']):
            orig_cwd = os.getcwd()
            try:
                os.chdir(self.tmpdir)
                mod.ROOT = self.tmpdir
                with self.assertRaises(FileExistsError):
                    mod.main()
            finally:
                os.chdir(orig_cwd)


# ---------------------------------------------------------------------------
# final_confirmation.py tests
# ---------------------------------------------------------------------------

class TestFinalConfirmation(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_campaign(self, with_accepted=True):
        binding = _make_binding(self.tmpdir)
        _make_deadline(self.tmpdir)
        _make_frozen_plan(self.tmpdir)
        evals = self.tmpdir / 'campaign/evaluations'
        evals.mkdir(parents=True, exist_ok=True)
        (self.tmpdir / 'campaign/candidates').mkdir(parents=True, exist_ok=True)
        outcomes = []
        if with_accepted:
            ev_dir = evals / 'ev001'
            ev_dir.mkdir()
            cand = self.tmpdir / 'campaign/candidates/frozen_ev001.py'
            cand.write_text('def run(*a): pass\n')
            import hashlib
            sha = hashlib.sha256(cand.read_bytes()).hexdigest()
            outcome = {
                'id': 'ev001', 'task': 'L1/069_rms_norm',
                'status': 'accepted',
                'conservative_geomean': 1.05,
                'completed_at_epoch': time.time() - 10,
                'candidate_source': str(cand.relative_to(self.tmpdir)),
                'source_sha256': sha,
                'cake_artifact_bindings': {},
                'handoff_row': {
                    'task': 'L1/069_rms_norm', 'status': 'accepted',
                    'reason': 'test', 'evidence': [],
                    'candidate_source': str(cand.relative_to(self.tmpdir)),
                    'correctness_source_sha256': sha,
                    'correctness_report': 'campaign/evaluations/ev001/correctness.json',
                    'latency_report': 'campaign/evaluations/ev001/latency.json',
                    'profile_skip_reason': 'test',
                    'cake_artifacts': [],
                },
            }
            (ev_dir / 'outcome.json').write_text(json.dumps(outcome))
        return binding

    def _load_module(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'final_confirmation_mod', ROOT / 'campaign/final_confirmation.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_missing_nominee_produces_missing_status(self):
        """final_confirmation produces status=missing_qualified_nominee when no accepted outcomes."""
        self._write_campaign(with_accepted=False)
        mod = self._load_module()
        orig_root = mod.ROOT
        orig_cwd = os.getcwd()
        try:
            os.chdir(self.tmpdir)
            mod.ROOT = self.tmpdir
            mod.main()
        finally:
            mod.ROOT = orig_root
            os.chdir(orig_cwd)
        result = json.loads((self.tmpdir / 'campaign/final-confirmation/result.json').read_text())
        self.assertEqual(result['status'], 'missing_qualified_nominee')
        self.assertIsNone(result['nominated_id'])

    def test_existing_final_confirmation_folder_raises(self):
        """final_confirmation raises FileExistsError if its folder already exists."""
        self._write_campaign(with_accepted=False)
        (self.tmpdir / 'campaign/final-confirmation').mkdir(parents=True)
        mod = self._load_module()
        orig_cwd = os.getcwd()
        try:
            os.chdir(self.tmpdir)
            mod.ROOT = self.tmpdir
            with self.assertRaises(FileExistsError):
                mod.main()
        finally:
            os.chdir(orig_cwd)

    def test_only_one_nominee_selected(self):
        """final_confirmation selects exactly the highest-geomean accepted outcome."""
        self._write_campaign(with_accepted=True)
        # Add a second (worse) accepted outcome
        evals = self.tmpdir / 'campaign/evaluations'
        ev_dir = evals / 'ev002'
        ev_dir.mkdir()
        cand = self.tmpdir / 'campaign/candidates/frozen_ev002.py'
        cand.write_text('def run(*a): pass\n')
        import hashlib
        sha = hashlib.sha256(cand.read_bytes()).hexdigest()
        outcome = {
            'id': 'ev002', 'task': 'L1/069_rms_norm',
            'status': 'accepted',
            'conservative_geomean': 0.98,
            'completed_at_epoch': time.time() - 5,
            'candidate_source': str(cand.relative_to(self.tmpdir)),
            'source_sha256': sha,
            'cake_artifact_bindings': {},
            'handoff_row': {'task': 'L1/069_rms_norm', 'status': 'accepted',
                            'reason': 'test', 'evidence': [],
                            'candidate_source': str(cand.relative_to(self.tmpdir)),
                            'correctness_source_sha256': sha,
                            'correctness_report': 'c/e.json', 'latency_report': 'c/l.json',
                            'profile_skip_reason': 'test', 'cake_artifacts': []},
        }
        (ev_dir / 'outcome.json').write_text(json.dumps(outcome))

        mod = self._load_module()
        orig_cwd = os.getcwd()
        try:
            os.chdir(self.tmpdir)
            mod.ROOT = self.tmpdir
            # Mock the actual GPU execution so we don't hang
            with patch.object(mod, '_acquire_device_lock', return_value=3), \
                 patch.object(mod, '_release_device_lock'), \
                 patch('subprocess.call', return_value=1):  # fail → confirmation_failed
                mod.main()
        finally:
            os.chdir(orig_cwd)

        result = json.loads((self.tmpdir / 'campaign/final-confirmation/result.json').read_text())
        # Should have selected ev001 (geomean=1.05 > 0.98)
        self.assertEqual(result.get('nominated_id'), 'ev001')


# ---------------------------------------------------------------------------
# bench_queue.py / CampaignOwner adapter tests
# ---------------------------------------------------------------------------

class TestBenchQueueAdapters(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _load_bench_queue(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'bench_queue_mod', ROOT / 'scripts/bench_queue.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_launcher_terminal_with_confirmed_result(self):
        """C500AuthorLauncher.terminal synthesises performance=measured on confirmed result."""
        mod = self._load_bench_queue()
        run = {
            'assignment_id': 'test-001-c-r1',
            'root': str(self.tmpdir),
            'device_id': 'c500-0',
            'physical_device': 0,
        }
        (self.tmpdir / 'campaign').mkdir(parents=True, exist_ok=True)
        (self.tmpdir / 'campaign/logs').mkdir(parents=True, exist_ok=True)
        (self.tmpdir / 'campaign/evaluations').mkdir(parents=True, exist_ok=True)
        (self.tmpdir / 'campaign/controller-status.json').write_text(
            json.dumps({'model_exit_code': 0, 'error': None, 'done_exists': False,
                        'owner_finalized_at_epoch': time.time()}))
        conf_dir = self.tmpdir / 'campaign/final-confirmation'
        conf_dir.mkdir(parents=True, exist_ok=True)
        (conf_dir / 'result.json').write_text(
            json.dumps({'status': 'confirmed', 'conservative_geomean': 1.03}))

        launcher = mod.C500AuthorLauncher()
        terminal = launcher.terminal(run, None, 0)
        self.assertEqual(terminal['performance']['status'], 'measured')
        self.assertEqual(terminal['execution']['status'], 'completed')
        self.assertEqual(terminal['confirmation']['status'], 'confirmed')

    def test_release_verifier_verified_on_clean_exit(self):
        """C500ReleaseVerifier marks release verified when exit_code is 0."""
        mod = self._load_bench_queue()
        verifier = mod.C500ReleaseVerifier()
        run = {'assignment_id': 'test-001', 'device_id': 'c500-0', 'physical_device': 0}
        terminal = {'assignment_id': 'test-001',
                    'execution': {'status': 'completed', 'exit_code': 0}}
        result = verifier.observe(run, {}, terminal)
        self.assertEqual(result['status'], 'verified')

    def test_release_verifier_unverified_on_failed_exit(self):
        """C500ReleaseVerifier marks release unverified on non-zero exit."""
        mod = self._load_bench_queue()
        verifier = mod.C500ReleaseVerifier()
        run = {'assignment_id': 'test-002', 'device_id': 'c500-0', 'physical_device': 0}
        terminal = {'assignment_id': 'test-002',
                    'execution': {'status': 'failed', 'exit_code': 1}}
        result = verifier.observe(run, {}, terminal)
        self.assertEqual(result['status'], 'unverified')

    def test_device_lock_acquired_and_released(self):
        """Device lock file is created and released without error (CPU-only, no GPU)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'evaluate_lock', ROOT / 'campaign/evaluate.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        uid = os.geteuid()
        lock = Path(f'/tmp/open-cake-ir-maca-{uid}-device-99.lock')
        fd = mod._acquire_device_lock(99)
        self.assertGreater(fd, 0)
        mod._release_device_lock(fd)
        self.assertTrue(lock.exists())
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
