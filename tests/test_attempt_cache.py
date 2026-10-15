"""Trusted attempt caches retain scope, filesystem policy and entry ordering."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from campaign.processes import candidate_sandbox, trusted_attempt_environment
from campaign.assay import seed_maca_helper

SOURCE = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == 'linux', 'Linux executable mount policy required')
class AttemptCacheTests(unittest.TestCase):
    def test_cache_roots_are_attempt_owned_and_preserve_runtime_visibility_and_tmp(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            original = {'TRITON_CACHE_DIR': '/tmp/triton', 'CUDA_VISIBLE_DEVICES': '0',
                        'LD_LIBRARY_PATH': '/captured/runtime', 'TMPDIR': '/tmp'}
            first, second = base / 'first', base / 'second'
            first.mkdir()
            second.mkdir()
            environments = [trusted_attempt_environment(original, output) for output in (first, second)]
            for output, environment in zip((first, second), environments):
                for variable in ('TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'TORCH_EXTENSIONS_DIR'):
                    self.assertTrue(Path(environment[variable]).is_relative_to(output))
                    self.assertTrue(Path(environment[variable]).is_dir())
                self.assertEqual(environment['TMPDIR'], '/tmp')
                self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '0')
                self.assertEqual(environment['LD_LIBRARY_PATH'], original['LD_LIBRARY_PATH'])
                self.assertEqual(trusted_attempt_environment(environment, output), environment)
            self.assertNotEqual(environments[0]['TRITON_CACHE_DIR'], environments[1]['TRITON_CACHE_DIR'])
            self.assertEqual(original['TRITON_CACHE_DIR'], '/tmp/triton')

    def test_noexec_output_and_redirected_cache_are_refused_before_use(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('campaign.processes.os.statvfs', return_value=types.SimpleNamespace(f_flag=os.ST_NOEXEC)):
                with self.assertRaisesRegex(PermissionError, 'executable attempt'):
                    trusted_attempt_environment({}, root)
            self.assertEqual(list(root.iterdir()), [])
            other = root / 'other'
            other.mkdir()
            (root / 'triton-cache').symlink_to(other, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'outside its attempt'):
                trusted_attempt_environment({}, root)

    def test_seed_copies_only_the_prepared_module_and_cannot_reuse_or_escape_an_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, output = root / 'prepared', root / 'attempt'
            (artifact / 'triton/sdk-key').mkdir(parents=True)
            module = artifact / 'triton/sdk-key/maca_utils.so'
            module.write_bytes(b'prepared SDK host module')
            (artifact / 'triton/sdk-key/previous_kernel.bin').write_bytes(b'must not inherit')
            (artifact / 'preparation.json').write_text(json.dumps({
                'cache': 'triton', 'module': 'triton/sdk-key/maca_utils.so'}))
            output.mkdir()
            environment = trusted_attempt_environment({}, output)
            cache = Path(environment['TRITON_CACHE_DIR'])
            seed = seed_maca_helper(artifact, cache, output)
            self.assertEqual(Path(seed['cache_module']).read_bytes(), module.read_bytes())
            self.assertEqual([path.name for path in cache.rglob('*') if path.is_file()], ['maca_utils.so'])
            with self.assertRaises(FileExistsError):
                seed_maca_helper(artifact, cache, output)
            elsewhere = root / 'outside'
            elsewhere.mkdir()
            with self.assertRaises(ValueError):
                seed_maca_helper(artifact, elsewhere, output)

    def test_candidate_namespace_does_not_receive_trusted_attempt_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, bundle, output = base / 'run', base / 'bundle', base / 'trusted-output'
            (root / 'campaign').mkdir(parents=True)
            bundle.mkdir()
            output.mkdir()
            (root / 'campaign/binding.json').write_text(json.dumps({'physical_device': 0}))
            environment = trusted_attempt_environment({}, output)
            protocol = {'deployment': {'bubblewrap': '/usr/bin/bwrap', 'assay_python': sys.executable,
                'control_device_node': '/dev/null', 'devices': {'0': {'render_node': '/dev/null'}}}}
            command = candidate_sandbox(root, SOURCE, bundle, protocol)
            for variable in ('TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'TORCH_EXTENSIONS_DIR'):
                self.assertNotIn(environment[variable], command)
            self.assertNotIn(str(output), command)

    def test_both_isolated_entry_actions_configure_caches_before_sdk_imports(self):
        if not os.statvfs(SOURCE).f_flag & os.ST_RDONLY:
            self.skipTest('read-only CPU deployment required for the real trusted entry')
        commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / 'root'
            (root / 'campaign').mkdir(parents=True)
            (root / 'campaign/binding.json').write_text(json.dumps({'bench_commit': commit}))
            for action in ('reference', 'assay'):
                output = base / action
                output.mkdir()
                request = output / 'request.json'
                request.write_text('{}')
                code = f'''import os, sys
sys.path.insert(0, {str(SOURCE)!r})
from pathlib import Path
from campaign import entry
from campaign import reference, assay
def observed(*args):
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    for name in ('TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'TORCH_EXTENSIONS_DIR'):
        assert Path(os.environ[name]).is_relative_to(Path({str(output)!r}))
    assert os.environ['TMPDIR'] == '/tmp'
    print('trusted entry cache setup precedes SDK imports')
reference.qualify_reference = observed
assay.evaluate = observed
sys.argv = ['entry', {action!r}, '--root', {str(root)!r}, '--state', {str(output)!r}, '--request', {str(request)!r}]
assert entry.main() == 0
'''
                result = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('precedes SDK imports', result.stdout)
