"""CPU preparation and cache-only use of the SDK host admission helper."""
import glob
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from campaign.assay import admit_device, prepare_maca_helper

SOURCE = Path(__file__).resolve().parents[1]


def metax_sdk_present():
    package = importlib.util.find_spec('triton')
    return package is not None and any((Path(path) / 'backends/metax/driver.py').is_file()
                                      for path in package.submodule_search_locations or [])


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class HelperBoundaryTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'linux', 'Linux executable mount flags required')
    def test_noexec_artifact_is_refused_before_any_compiler_or_device_import(self):
        with tempfile.TemporaryDirectory() as directory, patch('campaign.assay.validate_runtime_capture'), patch(
                'campaign.assay.os.statvfs', return_value=types.SimpleNamespace(f_flag=os.ST_NOEXEC)):
            with self.assertRaisesRegex(PermissionError, 'executable artifact'):
                prepare_maca_helper(Path(directory), Path(directory) / 'helper')

    def test_missing_helper_refuses_before_lease_or_device_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / 'campaign/binding.json', {'physical_device': 0, 'measurement_protocol': 'campaign/protocol.json'})
            write(root / 'campaign/protocol.json', {'deployment': {'devices': {'0': {
                'runtime_device': 0, 'expected_pci': '0000:0f:00'}}}})
            broker = types.ModuleType('open_cake_ir.evaluation.local_broker')
            observer = types.ModuleType('open_cake_ir.evaluation.triton_metax')
            broker.admit_local_job, observer.observe_local_metax = Mock(), Mock()
            modules = {name: types.ModuleType(name) for name in ('open_cake_ir', 'open_cake_ir.evaluation')}
            modules.update({broker.__name__: broker, observer.__name__: observer})
            with patch.dict(sys.modules, modules), patch('campaign.assay.validate_runtime_capture'):
                with self.assertRaises(FileNotFoundError):
                    admit_device(root, {'stop_at_epoch': float('inf')}, root,
                                 runtime_helper=root / 'missing-helper')
            broker.admit_local_job.assert_not_called()
            observer.observe_local_metax.assert_not_called()


@unittest.skipUnless(sys.platform == 'linux' and metax_sdk_present(), 'captured MetaX Linux SDK required')
class InstalledSdkHelperTests(unittest.TestCase):
    def test_actual_cpu_compile_load_and_fresh_cache_only_consumption_keep_tmp_noexec(self):
        self.assertFalse(glob.glob('/dev/dri/render*') or glob.glob('/dev/mxcd*'), 'CPU-only context required')
        self.assertTrue(os.statvfs('/tmp').f_flag & os.ST_NOEXEC, 'private /tmp must remain noexec')
        from importlib.metadata import version
        import torch
        from triton.runtime import build
        self.assertFalse(torch.cuda.is_initialized())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / '.deps/cake-ir/runtime/hosts/xcore1002.json', {'host_environment': {
                'python': {'version': platform.python_version()}, 'packages': {'triton': version('triton')}}})
            with patch('tempfile.tempdir', '/tmp'), patch(
                    'triton.runtime.build.subprocess.check_call', wraps=subprocess.check_call) as compiler:
                prepared = prepare_maca_helper(root, root / 'runtime-helper')
                self.assertEqual(compiler.call_count, 1)
                self.assertTrue(str(compiler.call_args.args[0][1]).startswith('/tmp/'))
            self.assertEqual(prepared['status'], 'host_helper_prepared')
            helper = root / 'runtime-helper'
            native = helper / prepared['module']
            self.assertTrue(native.is_file())
            self.assertFalse(os.statvfs(native).f_flag & os.ST_NOEXEC)
            code = f'''import sys
sys.path.insert(0, {str(SOURCE)!r})
from pathlib import Path
from unittest.mock import patch
from campaign.assay import prepared_maca_helper
from triton import knobs
from triton.backends.metax.driver import MacaUtils
from triton.runtime.build import compile_module_from_src
original_cache, original_build = knobs.cache.dir, knobs.build.impl
with patch('triton.runtime.build.subprocess.check_call', side_effect=AssertionError('compiler started in cache-only phase')):
    with prepared_maca_helper(Path({str(root)!r}), Path({str(helper)!r})) as loaded:
        assert loaded['build_policy'] == 'cache_only_no_build'
        MacaUtils()
        try:
            compile_module_from_src('missing helper module', 'must_not_build')
        except RuntimeError as error:
            assert 'prepare it without a device' in str(error), str(error)
        else:
            raise AssertionError('missing native helper must not compile')
assert knobs.cache.dir == original_cache
assert knobs.build.impl is original_build
print('fresh SDK helper cache load passed; missing helper build refused')
'''
            result = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('missing helper build refused', result.stdout)
        self.assertFalse(torch.cuda.is_initialized())
