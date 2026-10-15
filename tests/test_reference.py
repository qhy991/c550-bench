"""Fixed reference selection, canonical qualification and author isolation."""
import contextlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import c550bench
c550bench.upstream()
from campaign.binding import read_object
from campaign.processes import sandbox
from campaign.reference import compiled_original, prepare_reference, qualified_reference, qualify_reference
from campaign.supervisor import AssignmentSupervisor
try:
    from tests.test_campaign import repository, write
except ModuleNotFoundError:
    from test_campaign import repository, write

try:
    from tests.runtime_fixture import maca_protocol
except ModuleNotFoundError:
    from runtime_fixture import maca_protocol

SOURCE = Path(__file__).resolve().parents[1]


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        helper = patch('campaign.assay.prepare_maca_helper', return_value={'scope': 'CPU source fixture'})
        helper.start()
        self.addCleanup(helper.stop)

    def fixture(self, base, source='def run(x): return x.clone()\n'):
        root = base / 'run'
        task = {'id': 'L1/fixture', 'smoke_workload_uuid': 'fixture-0'}
        original = 'def run(x): return x.clone()\n'
        definition = {'name': 'identity', 'axes': {'n': {'type': 'var'}},
                      'inputs': {'x': {'shape': ['n'], 'dtype': 'float32'}},
                      'outputs': {'y': {'shape': ['n'], 'dtype': 'float32'}}, 'reference': original}
        baseline = 'baselines/selected.py'
        bench = base / 'bench'
        commit = repository(bench, {baseline: source, '.gitignore': '.data/\n.deps/\n'})
        write(root / 'suite.json', {'tasks': [task], 'correctness_rounds': 10, 'seed': 200,
                                   'target_arch': 'metax_c550'})
        write(root / 'sources.lock.json', {'fixture': 'pinned'})
        write(root / 'campaign/binding.json', {'task': task['id'], 'baseline': baseline,
              'bench_commit': commit, 'compiler_commit': 'fixture-compiler',
              'workload_ids': [f'fixture-{i}' for i in range(16)]})
        material = root / '.data/benchmark' / task['id']
        write(material / 'definition.json', definition)
        (material / 'reference.py').write_text(original)
        (material / 'workload.jsonl').write_text(''.join(json.dumps({
            'uuid': f'fixture-{index}', 'axes': {'n': 4}, 'inputs': {'x': {'type': 'random'}}
        }) + '\n' for index in range(16)))
        return root, bench

    def test_frozen_source_is_staged_privately_once_and_changed_source_refuses(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(c550bench, 'upstream'):
            base = Path(directory).resolve()
            root, bench = self.fixture(base)
            bundle = base / 'private/selected'
            prepared = prepare_reference(root, bench, bundle)
            self.assertEqual(prepared['baseline'], 'baselines/selected.py')
            self.assertEqual((bundle / 'adapter.py').read_text(), 'def run(x): return x.clone()\n')
            self.assertFalse(bundle.is_relative_to(root))
            with self.assertRaises(FileExistsError):
                prepare_reference(root, bench, bundle)
            (bench / 'baselines/selected.py').write_text('def run(x): return x * 0\n')
            with self.assertRaisesRegex(ValueError, 'not clean'):
                prepare_reference(root, bench, base / 'private/changed')

    def check_fixture(self, source):
        import torch
        with tempfile.TemporaryDirectory() as directory, patch.object(c550bench, 'upstream'):
            base = Path(directory).resolve()
            root, bench = self.fixture(base, 'import torch\n' + source)
            bundle = base / 'private/selected'
            prepare_reference(root, bench, bundle)
            output = base / 'qualification'
            output.mkdir()
            original_check = c550bench.check_problem
            def cpu_check(*args, **kwargs):
                self.assertEqual(kwargs['device'], 'cuda:0')
                self.assertEqual(kwargs['workload_scope'], 'all')
                self.assertEqual(kwargs['rounds'], 10)
                kwargs['device'] = 'cpu'
                return original_check(*args, **kwargs)
            with patch('campaign.assay.admit_device'), patch.object(c550bench, 'check_problem', cpu_check):
                result = qualify_reference(root, {'reference_bundle': str(bundle),
                    'stop_at_epoch': time.time() + 60}, output)
            return result, read_object(output / 'correctness.json')

    def test_actual_selected_callable_traverses_all_original_cases_without_device_claim(self):
        result, correctness = self.check_fixture('def run(x): return x.clone()\n')
        self.assertEqual(correctness['status'], 'passed')
        self.assertEqual(len(correctness['cases']), 160)
        self.assertFalse(result['correctness']['full_device_correctness'])
        self.assertEqual(result['performance'], 'coverage_missing')
        self.assertEqual(result['measurement']['status'], 'coverage_missing')
        self.assertEqual(result['profile']['status'], 'coverage_missing')

    def test_wrong_selected_callable_fails_the_canonical_comparator(self):
        result, correctness = self.check_fixture('def run(x): return x * 0\n')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(correctness['status'], 'failed')
        self.assertEqual(correctness['cases'][0]['workload_uuid'], 'fixture-0')
        self.assertFalse(correctness['cases'][0]['passed'])

    def test_candidate_intake_refuses_unqualified_or_different_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root, _ = self.fixture(base)
            source = read_object(root / 'campaign/binding.json')
            receipt = base / 'receipt.json'
            write(receipt, {'source': source, 'status': 'correct',
                            'correctness': {'full_device_correctness': False}})
            with self.assertRaisesRegex(ValueError, 'complete original device correctness'):
                qualified_reference(root, receipt)
            source['baseline'] = 'another.py'
            write(receipt, {'source': source, 'status': 'correct',
                            'correctness': {'full_device_correctness': True}})
            with self.assertRaisesRegex(ValueError, 'frozen selection'):
                qualified_reference(root, receipt)


class ReferenceLifecycleTests(unittest.TestCase):
    def supervisor(self, base):
        owner = object.__new__(AssignmentSupervisor)
        owner.root, owner.state = base / 'run', base / 'state'
        owner.folder = owner.state / 'assignments/one'
        owner.identity = 'one'
        owner.binding = {'bench_commit': 'fixture-bench', 'compiler_commit': 'fixture-compiler'}
        owner.claim = {'search_stop_at_epoch': time.time() + 60}
        owner.results = []
        owner.search_closed = False
        owner.binding_with_commit = lambda: 'fixture-prepared'
        owner.service = lambda: None
        return owner

    def test_one_reference_opportunity_precedes_author_and_failure_prevents_author(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                owner = self.supervisor(Path(directory))
                calls = []
                def reference():
                    calls.append('reference')
                    if fail:
                        raise RuntimeError('reference failed; opportunity consumed')
                def author():
                    calls.append('author')
                    return 0
                owner.qualify_reference, owner.author = reference, author
                with patch('campaign.supervisor.clean_checkout'):
                    result = owner.run()
                self.assertEqual(calls, ['reference'] if fail else ['reference', 'author'])
                self.assertEqual(result['performance']['status'], 'coverage_missing')
                if fail:
                    self.assertIn('opportunity consumed', result['execution']['reason'])


@contextlib.contextmanager
def cpu_graph_backend():
    from torch._dynamo.backends.registry import lookup_backend
    def lookup(name):
        return (lambda graph, inputs: graph.forward) if name == 'inductor' else lookup_backend(name)
    with patch('torch._dynamo.backends.registry.lookup_backend', side_effect=lookup):
        yield


class CompilationTests(unittest.TestCase):
    def test_file_loaded_original_captures_graph_with_module_global(self):
        import torch
        import torch._dynamo
        name = 'c550bench_test_compiled_original'
        previous = {name: sys.modules[name]} if name in sys.modules else {}
        def restore():
            sys.modules.pop(name, None)
            sys.modules.update(previous)
            torch._dynamo.reset()
        self.addCleanup(restore)
        torch._dynamo.reset()
        with tempfile.TemporaryDirectory() as directory, cpu_graph_backend():
            root = Path(directory)
            path = root / 'original.py'
            path.write_text('import torch\nOFFSET = torch.tensor(2.)\ndef run(value):\n    return value + OFFSET\n')
            original = c550bench.load_module(path, name)
            route = {'kind': 'inductor', 'capture': 'fullgraph', 'dynamic': False,
                     'known_eager_regions': []}
            call, diagnostics = compiled_original(original, route, root)
            self.assertTrue(torch.equal(call(torch.ones(4)), torch.full((4,), 3.)))
            self.assertEqual(len(diagnostics()['graphs']), 1)
            self.assertEqual(diagnostics()['graph_breaks'], {})
            self.assertTrue((root / 'graph-0000.py').is_file())

    def compile(self, root, function, capture='fullgraph'):
        import torch._dynamo
        torch._dynamo.reset()
        route = {'kind': 'inductor', 'capture': capture, 'dynamic': False,
                 'known_eager_regions': ['explicit CPU test graph break'] if capture == 'declared_partial' else []}
        return compiled_original(types.SimpleNamespace(run=function), route, root)

    def test_strict_actual_dynamo_refuses_graph_break_without_eager_replacement(self):
        import torch
        def function(value):
            torch._dynamo.graph_break()
            return value + 1
        with tempfile.TemporaryDirectory() as directory, cpu_graph_backend():
            call, diagnostics = self.compile(Path(directory), function)
            with self.assertRaises(torch._dynamo.exc.Unsupported):
                call(torch.ones(4))
            self.assertEqual(diagnostics()['graphs'], [])

    def test_recompile_exhaustion_is_a_hard_failure_and_partial_breaks_are_recorded(self):
        import torch
        import torch._dynamo.config as config
        with tempfile.TemporaryDirectory() as directory, cpu_graph_backend():
            call, diagnostics = self.compile(Path(directory), lambda value: value + 1)
            with config.patch(recompile_limit=1):
                self.assertTrue(torch.equal(call(torch.ones(4)), torch.full((4,), 2.)))
                with self.assertRaises(Exception) as failure:
                    call(torch.ones(5))
                self.assertIn('recompile_limit', str(failure.exception))
            self.assertTrue(diagnostics()['fallback_policy']['fail_on_recompile_limit_hit'])
        def partial(value):
            first = value + 1
            torch._dynamo.graph_break()
            return first * 2
        with tempfile.TemporaryDirectory() as directory, cpu_graph_backend():
            call, diagnostics = self.compile(Path(directory), partial, 'declared_partial')
            self.assertTrue(torch.equal(call(torch.ones(4)), torch.full((4,), 4.)))
            self.assertTrue(diagnostics()['graph_breaks'])
            self.assertFalse(diagnostics()['uniform_compilation_claim'])


class AuthorMountTests(unittest.TestCase):
    def test_both_source_and_run_hide_adapter_trees_selected_file_and_git_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root, source = base / 'run', base / 'source'
            for tree in (root, source):
                for name in ('baselines', 'examples', '.git'):
                    (tree / name).mkdir(parents=True)
                (tree / 'chosen.py').write_text('secret = 1\n')
                (tree / 'other-task.py').write_text('sibling performance source = 1\n')
                (tree / 'campaign/__pycache__').mkdir(parents=True)
                (tree / 'campaign/reference.py').write_text('secret = 1\n')
                (tree / 'campaign/__pycache__/reference.pyc').write_bytes(b'secret')
            write(root / 'campaign/binding.json', {'baseline': 'chosen.py'})
            write(root / 'campaign/frozen-plan.json', {'baseline_files': {
                'L1/fixture': 'chosen.py', 'L2/other': 'other-task.py'}})
            protocol = maca_protocol(base, bubblewrap='/usr/bin/bwrap', hmz_root='/hmz',
                hmz_python_root='/hmz-python', provider_native='/provider')
            command = sandbox(root, source, writable=[], protocol=protocol, author=True)
            for tree in (root, source):
                for name in ('baselines', 'examples', '.git'):
                    index = command.index(str(tree / name))
                    self.assertEqual(command[index - 1], '--tmpfs')
                for name in ('chosen.py', 'other-task.py'):
                    index = command.index(str(tree / name))
                    self.assertEqual(command[index - 2:index], ['--ro-bind', '/dev/null'])

    @unittest.skipUnless(sys.platform == 'linux' and shutil.which('bwrap'), 'Linux bubblewrap required')
    def test_real_author_namespace_hides_both_performance_sources_and_preserves_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root, source = base / 'run', base / 'source'
            for tree in (root, source):
                for name in ('baselines', 'examples', '.git'):
                    (tree / name).mkdir(parents=True)
                    (tree / name / 'private.py').write_text('secret = 1\n')
                (tree / 'chosen.py').write_text('secret = 1\n')
                (tree / 'other-task.py').write_text('sibling performance source = 1\n')
                (tree / 'campaign/__pycache__').mkdir(parents=True)
                (tree / 'campaign/reference.py').write_text('secret = 1\n')
                (tree / 'campaign/__pycache__/reference.pyc').write_bytes(b'secret')
            write(root / 'campaign/binding.json', {'baseline': 'chosen.py'})
            write(root / 'campaign/frozen-plan.json', {'baseline_files': {
                'L1/fixture': 'chosen.py', 'L2/other': 'other-task.py'}})
            write(root / '.data/benchmark/L1/fixture/definition.json', {'spec': 'allowed'})
            private = base / 'owner/reference-source'
            private.mkdir(parents=True)
            (private / 'adapter.py').write_text('secret = 1\n')
            script = root / 'probe.py'
            script.write_text(
                'import json\nfrom pathlib import Path\n'
                'def source_denied(path):\n'
                '    try:\n'
                '        value = Path(path).read_bytes()\n'
                '    except PermissionError:\n'
                '        return\n'
                '    assert value == b"", path\n' + ''.join(
                f'assert not Path({str(tree / name / "private.py")!r}).exists()\n'
                for tree in (root, source) for name in ('baselines', 'examples', '.git')) + ''.join(
                f'source_denied({str(tree / "chosen.py")!r})\n'
                for tree in (root, source)) + ''.join(
                f'source_denied({str(tree / "other-task.py")!r})\n'
                for tree in (root, source)) + ''.join(
                f'source_denied({str(tree / "campaign/reference.py")!r})\n'
                f'assert not Path({str(tree / "campaign/__pycache__/reference.pyc")!r}).exists()\n'
                for tree in (root, source)) +
                f'assert not Path({str(private)!r}).exists()\n' +
                f'assert Path({str(root / ".data/benchmark/L1/fixture/definition.json")!r}).is_file()\n' +
                f'assert json.loads(Path({str(root / ".data/benchmark/L1/fixture/definition.json")!r}).read_text()) == {{"spec": "allowed"}}\n')
            protocol = maca_protocol(base, bubblewrap=shutil.which('bwrap'), hmz_root='/absent-hmz',
                hmz_python_root='/absent-python', provider_native='/absent-provider')
            command = sandbox(root, source, writable=[], protocol=protocol, author=True)
            result = subprocess.run([*command, sys.executable, '-I', str(script)],
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0,
                             f'author mask probe failed\nstdout={result.stdout}\nstderr={result.stderr}')


if __name__ == '__main__':
    unittest.main()
