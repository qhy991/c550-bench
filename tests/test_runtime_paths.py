"""SDK root exposure and preflight refusal without device access."""
import copy
import errno
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from campaign.assay import MACA_BLAS_FILES, admit_device, prepare_maca_helper, validate_runtime_capture
from campaign.processes import admit_private_paths, candidate_sandbox, maca_runtime_paths, runtime_paths, sandbox
try:
    from tests.runtime_fixture import maca_protocol
except ModuleNotFoundError:
    from runtime_fixture import maca_protocol

SOURCE = Path(__file__).resolve().parents[1]


class RuntimePathsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.protocol = maca_protocol(self.base, bubblewrap='/usr/bin/bwrap', assay_python=sys.executable,
            hmz_root='/absent-hmz', hmz_python_root='/absent-python', provider_native='/absent-provider',
            control_device_node='/dev/null', devices={'0': {'render_node': '/dev/null'}})
        self.maca = maca_runtime_paths(self.protocol)
        self.required = [Path(self.protocol['deployment'][key]) for key in ('runtime_library', 'activity_library')]
        self.required.extend(self.maca.alias / name for name in MACA_BLAS_FILES)
        (self.maca.root / 'lib').mkdir()
        for path in self.required:
            path.write_bytes(b'CPU path fixture')
        self.root = self.base / 'run'
        self.write('campaign/binding.json', {'physical_device': 0, 'baseline': 'chosen.py', 'task': 'fixture',
                                           'measurement_protocol': 'campaign/protocol.json'})
        self.write('campaign/protocol.json', self.protocol)
        self.write('campaign/frozen-plan.json', {'baseline_files': {'fixture': 'chosen.py'}})
        self.capture = {'target': 'xcore1002', 'host_environment': {'kind': 'maca',
            'runtime': {'build_environment': {'MACA_PATH': str(self.maca.alias)}},
            'python': {'version': platform.python_version()}, 'packages': {}}}
        self.capture_path = '.deps/cake-ir/runtime/hosts/xcore1002.json'
        self.write(self.capture_path, self.capture)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_both_sandbox_builders_mount_canonical_root_before_alias_readonly(self):
        paths = runtime_paths(self.protocol)
        self.assertEqual(paths[-2:], (str(self.maca.root), str(self.maca.alias)))
        self.assertNotIn(str(self.base), paths)
        self.assertNotIn('/opt/maca', paths)
        commands = [sandbox(self.root, SOURCE, writable=[], protocol=self.protocol, author=author)
                    for author in (False, True)]
        commands.append(candidate_sandbox(self.root, SOURCE, self.base / 'bundle', self.protocol))
        for command in commands:
            mounted = [command[i + 1:i + 3] for i, arg in enumerate(command) if arg == '--ro-bind']
            root = [str(self.maca.root)] * 2
            alias = [str(self.maca.alias)] * 2
            self.assertLess(mounted.index(root), mounted.index(alias))
            self.assertNotIn([str(self.base)] * 2, mounted)
        protocol = copy.deepcopy(self.protocol)
        for key in ('runtime_library', 'activity_library'):
            protocol['deployment'][key] = str(self.maca.root / Path(protocol['deployment'][key]).relative_to(self.maca.alias))
        self.assertEqual(runtime_paths(protocol).count(str(self.maca.root)), 1)

    def test_private_paths_under_alias_or_canonical_sdk_are_refused(self):
        for parent in (self.maca.alias, self.maca.root):
            with self.subTest(parent=parent), self.assertRaisesRegex(ValueError, 'overlaps untrusted runtime'):
                admit_private_paths([parent / 'private-evidence'], self.protocol, SOURCE)
        admit_private_paths([self.base / 'private-evidence'], self.protocol, SOURCE)

    def test_relative_divergent_or_non_direct_library_paths_are_refused(self):
        for value in ('relative/lib/runtime.so', str(self.base / 'other/lib/runtime.so'),
                      str(self.maca.alias / 'lib/subdir/runtime.so'), str(self.maca.alias / '../lib/runtime.so')):
            protocol = copy.deepcopy(self.protocol)
            protocol['deployment']['runtime_library'] = value
            with self.subTest(path=value), self.assertRaisesRegex(ValueError, 'one absolute SDK lib directory'):
                maca_runtime_paths(protocol)

    def test_missing_root_is_not_silently_omitted_and_broad_or_file_root_is_refused(self):
        self.maca.alias.unlink()
        with self.assertRaises(FileNotFoundError):
            runtime_paths(self.protocol)
        for target in (self.base, Path('/'), self.required[0].relative_to(self.maca.alias)):
            if not target.is_absolute():
                target = self.maca.root / target
            self.maca.alias.symlink_to(target)
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'MACA SDK root must be a directory'):
                runtime_paths(self.protocol)
            self.maca.alias.unlink()

    def test_capture_accepts_alias_or_canonical_root_and_preserves_package_checks(self):
        validate_runtime_capture(self.root)
        self.capture['host_environment']['runtime']['build_environment']['MACA_PATH'] = str(self.maca.root)
        self.write(self.capture_path, self.capture)
        validate_runtime_capture(self.root)
        self.capture['host_environment']['python']['version'] = 'different'
        self.write(self.capture_path, self.capture)
        with self.assertRaisesRegex(ValueError, 'assay Python differs'):
            validate_runtime_capture(self.root)
        self.capture['host_environment']['python']['version'] = platform.python_version()
        self.capture['host_environment']['packages'] = {'fixture-package': 'expected'}
        self.write(self.capture_path, self.capture)
        with patch('campaign.assay.version', return_value='different'), self.assertRaisesRegex(ValueError, 'assay fixture-package differs'):
            validate_runtime_capture(self.root)

    def test_disagreeing_capture_and_wrong_target_are_refused(self):
        environment = self.capture['host_environment']['runtime']['build_environment']
        for value in (str(self.base), 'relative-sdk'):
            environment['MACA_PATH'] = value
            self.write(self.capture_path, self.capture)
            with self.subTest(path=value), self.assertRaisesRegex(ValueError, 'SDK root differs'):
                validate_runtime_capture(self.root)
        del environment['MACA_PATH']
        self.write(self.capture_path, self.capture)
        with self.assertRaisesRegex(ValueError, 'no MACA_PATH'):
            validate_runtime_capture(self.root)
        self.capture['target'] = 'other'
        self.write(self.capture_path, self.capture)
        with self.assertRaisesRegex(ValueError, 'no exact MACA host capture'):
            validate_runtime_capture(self.root)

    def test_each_missing_required_file_refuses_before_helper_creation_or_lease_import(self):
        original_import = __import__
        def guarded_import(name, *args, **kwargs):
            if name == 'triton' or name.startswith(('triton.', 'open_cake_ir.')):
                self.fail('SDK or broker import preceded preflight refusal: ' + name)
            return original_import(name, *args, **kwargs)
        for path in self.required:
            path.unlink()
            output = self.root / 'not-created'
            with self.subTest(path=path), patch('builtins.__import__', side_effect=guarded_import), patch(
                    'campaign.assay.seed_maca_helper') as seed:
                with self.assertRaises(FileNotFoundError) as failure:
                    prepare_maca_helper(self.root, output)
                self.assertEqual(Path(failure.exception.filename).name, path.name)
                self.assertFalse(output.exists())
                with self.assertRaises(FileNotFoundError):
                    admit_device(self.root, {}, output, runtime_helper=self.root / 'not-prepared')
                seed.assert_not_called()
            path.write_bytes(b'CPU path fixture')

    def test_escaped_and_nonregular_runtime_files_are_refused(self):
        path = self.required[-1]
        path.unlink()
        outside = self.base / 'outside-module'
        outside.write_bytes(b'outside')
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'outside the selected SDK root'):
            validate_runtime_capture(self.root)
        path.unlink()
        path.mkdir()
        with self.assertRaisesRegex(ValueError, 'regular file'):
            validate_runtime_capture(self.root)
        path.rmdir()
        os.mkfifo(path)
        with self.assertRaisesRegex(ValueError, 'regular file'):
            validate_runtime_capture(self.root)

    def test_read_error_preserves_path_without_assuming_role_dac_capabilities(self):
        path = self.required[-1]
        original_open = Path.open
        def deny_module(candidate, *args, **kwargs):
            if candidate == path:
                raise PermissionError(errno.EACCES, 'software injected read denial', str(path))
            return original_open(candidate, *args, **kwargs)
        with patch.object(Path, 'open', autospec=True, side_effect=deny_module):
            with self.assertRaises(PermissionError) as failure:
                validate_runtime_capture(self.root)
        self.assertEqual(failure.exception.filename, str(path))

    @unittest.skipUnless(sys.platform == 'linux' and shutil.which('bwrap'), 'Linux bubblewrap required')
    def test_real_candidate_namespace_reads_both_sdk_spellings_readonly_and_hides_private_siblings(self):
        bundle = self.base / 'bundle'
        bundle.mkdir()
        private = self.base / 'private-evidence'
        private.write_bytes(b'private')
        code = 'import errno\nfrom pathlib import Path\n'
        for alias in self.required:
            canonical = self.maca.root / alias.relative_to(self.maca.alias)
            code += f'a, b = Path({str(alias)!r}), Path({str(canonical)!r})\n'
            code += 'assert a.read_bytes() == b.read_bytes() == b"CPU path fixture"\nassert a.samefile(b)\n'
            code += 'for path in (a, b):\n    try:\n        path.write_bytes(b"changed")\n'
            code += '    except OSError as error:\n        assert error.errno == errno.EROFS, error\n'
            code += '    else:\n        raise AssertionError("SDK mount was writable")\n'
        code += f'assert not Path({str(private)!r}).exists()\n'
        command = candidate_sandbox(self.root, SOURCE, bundle, self.protocol)
        result = subprocess.run([*command[:-6], sys.executable, '-I', '-c', code],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
