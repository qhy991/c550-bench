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

from campaign.assay import admit_device, prepare_maca_helper, seed_maca_helper
from campaign.processes import trusted_attempt_environment

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
            with patch.dict(sys.modules, modules), patch('campaign.assay.validate_runtime_capture'), patch.dict(
                    os.environ, {'TRITON_CACHE_DIR': str(root / 'cache')}):
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
            write(root / 'campaign/binding.json', {'measurement_protocol': 'campaign/protocol.json'})
            write(root / 'campaign/protocol.json', json.loads((SOURCE / 'campaign/protocol.json').read_text()))
            write(root / '.deps/cake-ir/runtime/hosts/xcore1002.json', {'target': 'xcore1002',
                'host_environment': {'kind': 'maca', 'runtime': {
                    'build_environment': {'MACA_PATH': os.environ['MACA_PATH']}},
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
            output = root / 'attempt'
            output.mkdir()
            environment = trusted_attempt_environment(dict(os.environ), output)
            seeded = seed_maca_helper(helper, Path(environment['TRITON_CACHE_DIR']), output)
            native_source = '''#include <Python.h>
static PyObject *value(PyObject *a, PyObject *b) { return PyLong_FromLong(17); }
static PyMethodDef methods[] = {{"value", value, METH_NOARGS, NULL}, {NULL, NULL, 0, NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "reference_cache_probe", NULL, -1, methods};
PyMODINIT_FUNC PyInit_reference_cache_probe(void) { return PyModule_Create(&module); }
'''
            code = f'''import os, sys
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
sys.path.insert(0, {str(SOURCE)!r})
from pathlib import Path
from unittest.mock import Mock, patch
from campaign.assay import prepared_maca_helper, _helper_module
from triton import knobs
from triton.backends.metax.driver import MacaUtils
from triton.runtime.build import compile_module_from_src
original_cache, original_build = knobs.cache.dir, knobs.build.impl
assert original_cache == os.environ['TRITON_CACHE_DIR']
with patch('triton.runtime.build.subprocess.check_call', side_effect=AssertionError('compiler started in cache-only phase')):
    assert str(_helper_module(MacaUtils())) == {seeded['cache_module']!r}
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
from torch._inductor.runtime.cache_dir_utils import cache_dir, triton_cache_dir
assert cache_dir() == os.environ['TORCHINDUCTOR_CACHE_DIR']
assert triton_cache_dir(0) == os.environ['TRITON_CACHE_DIR']
from torch.utils.cpp_extension import _get_build_directory
assert Path(_get_build_directory('cpu_reference_probe', False)).is_relative_to(Path(os.environ['TORCH_EXTENSIONS_DIR']))
source = {native_source!r}
native = compile_module_from_src(source, 'reference_cache_probe')
assert native.value() == 17
assert Path(native.__file__).is_relative_to(Path(os.environ['TRITON_CACHE_DIR']))
from concurrent.futures import Future
from types import SimpleNamespace
from torch._inductor import async_compile as asynchronous
pool = Mock()
pool.submit.return_value = Future()
with patch.object(asynchronous.AsyncCompile, 'use_process_pool', return_value=True), patch.object(asynchronous.AsyncCompile, 'process_pool', return_value=pool), patch.object(asynchronous.CompiledTritonKernels, 'get', return_value=None), patch.object(asynchronous.CompiledTritonKernels, 'save'), patch.object(asynchronous, 'V', SimpleNamespace(graph=SimpleNamespace(is_backward=False))):
    asynchronous.AsyncCompile().triton('never_compiled', '# CPU forwarding probe only')
extra_env = pool.submit.call_args.args[2]
assert extra_env == {{key: os.environ[key] for key in ('TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR')}}
from torch._inductor.runtime.compile_tasks import _worker_compile_triton
class Kernel:
    def precompile(self, **kwargs):
        assert cache_dir() == extra_env['TORCHINDUCTOR_CACHE_DIR']
        assert triton_cache_dir(0) == extra_env['TRITON_CACHE_DIR']
    def prepare_for_pickle(self): pass
with patch.dict(os.environ, {{'TRITON_CACHE_DIR': '/tmp/wrong-triton', 'TORCHINDUCTOR_CACHE_DIR': '/tmp/wrong-inductor'}}):
    _worker_compile_triton(lambda: Kernel(), extra_env, {{}})
print('fresh SDK seeded cache load and native extension passed; missing helper build refused; async cache forwarding passed')
'''
            result = subprocess.run([sys.executable, '-I', '-c', code], env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('missing helper build refused', result.stdout)
        self.assertFalse(torch.cuda.is_initialized())
