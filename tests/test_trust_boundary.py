"""Adversarial software probes for candidate, deadline and frozen-plan boundaries."""
import contextlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import c550bench
c550bench.upstream()
import torch
from sol_execbench.core.data import Definition, ToleranceSpec

from campaign.binding import reconcile_allocation
from campaign.isolated import CandidateSession
from campaign.processes import ProcessIdentity, candidate_sandbox, run_bounded
from campaign.wire import decode, encode

SOURCE = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def child(root, source):
    bundle = root / 'bundle'
    bundle.mkdir()
    (bundle / 'candidate.py').write_text(source)
    output = root / 'worker-evidence'
    output.mkdir()
    with patch('campaign.isolated.parent_death_signal'), (patch.object(
            ProcessIdentity, 'observe', side_effect=lambda pid: ProcessIdentity(pid, 'cpu-test'))
            if sys.platform != 'linux' else contextlib.nullcontext()):
        with CandidateSession([sys.executable, '-I', str(SOURCE / 'campaign/worker.py'),
                               str(bundle), 'cpu', str(SOURCE)],
                              environment=dict(os.environ), directory=output,
                              stop=time.time() + 30, hard_stop=time.time() + 32,
                              maximum_output_bytes=1024, device='cpu') as run:
            yield run


class IsolationTests(unittest.TestCase):
    def test_wrong_output_cannot_patch_trusted_comparison_or_measurement(self):
        definition = Definition.model_validate({
            'name': 'identity', 'axes': {'n': {'type': 'var'}},
            'inputs': {'x': {'shape': ['n'], 'dtype': 'float32'}},
            'outputs': {'y': {'shape': ['n'], 'dtype': 'float32'}},
            'reference': 'def run(x): return x.clone()\n'})
        source = f'''import sys
sys.path.insert(0, {str(SOURCE)!r})
import c550bench, campaign.assay, campaign.owner
c550bench.compare_outputs = lambda *args: {{"passed": True}}
campaign.assay.paired_summary = lambda *args: {{"quality_passed": True, "candidate_over_baseline": 0.001}}
campaign.owner.publish_new = lambda *args: None
def run(x): return x * 0
'''
        with tempfile.TemporaryDirectory() as directory:
            with child(Path(directory), source) as run:
                actual = run(torch.ones(4))
            result = c550bench.compare_outputs(actual, torch.ones(4), definition,
                                               ToleranceSpec(), {'n': 4})
            self.assertFalse(result['passed'])
            self.assertEqual(campaign_coverage()['status'], 'coverage_missing')

    def test_forged_result_and_success_exit_are_not_a_tensor_result(self):
        source = '''import json, os, sys
from pathlib import Path
Path(sys.argv[1], 'result.json').write_text(json.dumps({'id':'one','phase':'search','candidate':'a','status':'passed'}))
os._exit(0)
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with child(root, source) as run:
                with self.assertRaises((RuntimeError, BrokenPipeError)):
                    run(torch.ones(4))
                self.assertIsNotNone(run.failure)
            self.assertFalse((root / 'worker-evidence/result.json').exists())

    def test_complete_stateful_callable_retains_ten_fresh_rounds_per_uuid(self):
        source = '''previous = None
calls = 0
def run(x):
    global previous, calls
    calls += 1
    if previous is not None and x.equal(previous):
        raise RuntimeError('reused input')
    previous = x.clone()
    return x.clone() if calls <= 160 else x * 0
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {'id': 'L1/identity', 'smoke_workload_uuid': 'fixture-0'}
            problem = root / '.data/benchmark' / task['id']
            problem.mkdir(parents=True)
            definition = Definition.model_validate({
                'name': 'identity', 'axes': {'n': {'type': 'var'}},
                'inputs': {'x': {'shape': ['n'], 'dtype': 'float32'}},
                'outputs': {'y': {'shape': ['n'], 'dtype': 'float32'}},
                'reference': 'def run(x): return x.clone()\n'}).model_dump(mode='json')
            (problem / 'definition.json').write_text(json.dumps(definition))
            (problem / 'reference.py').write_text(definition['reference'])
            (problem / 'workload.jsonl').write_text(''.join(json.dumps({
                'uuid': f'fixture-{index}', 'axes': {'n': 4}, 'inputs': {'x': {'type': 'random'}}
            }) + '\n' for index in range(16)))
            (root / 'sources.lock.json').write_text('{}')
            (root / 'suite.json').write_text(json.dumps({'correctness_rounds': 10, 'target_arch': 'metax_c550'}))
            with child(root, source) as run, patch.object(c550bench, 'ROOT', root):
                result = c550bench.check_problem(task, device='cpu', candidate_path=root / 'bundle/candidate.py',
                    symbol='run', reference_selfcheck=False, workload_scope='all', rounds=10,
                    output=root / 'correctness.json', seed=200, threads=1, candidate_callable=run)
                self.assertEqual(result['status'], 'passed')
                self.assertEqual(len(result['cases']), 160)
                self.assertEqual(len({case['workload_uuid'] for case in result['cases']}), 16)
                self.assertFalse(result['full_device_correctness'])
                self.assertTrue(torch.equal(run(torch.ones(4)), torch.zeros(4)))

    def test_tensor_transport_preserves_input_strides_and_output_types(self):
        inputs = [torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).t(), 7, 0.25, None]
        frame = encode(inputs)
        decoded = decode(frame[8:], device='cpu', preserve_strides=True)
        self.assertEqual(decoded[0].stride(), inputs[0].stride())
        self.assertTrue(torch.equal(decoded[0], inputs[0]))
        self.assertEqual(decoded[1:], inputs[1:])
        with self.assertRaises(ValueError):
            decode(b'\x00\x00\x00\x02{}', device='cpu')

    def test_worker_mounts_hide_live_helpers_requests_results_and_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'campaign').mkdir()
            (root / 'campaign/binding.json').write_text(json.dumps({'physical_device': 0}))
            protocol = {'deployment': {'bubblewrap': '/usr/bin/bwrap', 'assay_python': '/opt/conda/bin/python',
                        'control_device_node': '/dev/mxcd', 'devices': {'0': {'render_node': '/dev/dri/renderD128'}}}}
            command = candidate_sandbox(root, SOURCE, root / 'saved', protocol)
            mounted = [command[i + 1] for i, arg in enumerate(command) if arg in ('--ro-bind', '--bind')]
            self.assertNotIn(str(root), mounted)
            self.assertNotIn(str(SOURCE), mounted)
            for name in ('campaign/candidates', 'campaign/schedules', 'campaign/compiler-actions',
                         'campaign/requests', 'campaign/responses', '.data'):
                self.assertFalse(any(str(root / name).startswith(path + '/') or str(root / name) == path
                                     for path in mounted))
            self.assertIn('--unshare-pid', command)
            self.assertIn('--unshare-net', command)


    @unittest.skipUnless(sys.platform == 'linux' and shutil.which('bwrap'),
                         'actual namespace isolation requires Linux bubblewrap')
    def test_real_namespace_refuses_live_absolute_helper_and_result_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'campaign').mkdir()
            (root / 'campaign/binding.json').write_text(json.dumps({'physical_device': 0}))
            (root / '.deps/cake-ir').mkdir(parents=True)
            live = root / 'campaign/candidates/a'
            live.mkdir(parents=True)
            (live / 'helper.py').write_text('value = 99\n')
            bundle = root / 'saved'
            bundle.mkdir()
            (bundle / 'helper.py').write_text('def run(x): return x.clone()\n')
            evidence = root / 'evidence'
            evidence.mkdir()
            forbidden = evidence / 'result.json'
            (bundle / 'candidate.py').write_text(
                'from pathlib import Path\nfrom helper import run as saved\n'
                'def run(x):\n'
                f'    if Path({str(live / "helper.py")!r}).exists(): return x * 0\n'
                '    try:\n'
                f'        Path({str(forbidden)!r}).write_text("forged")\n'
                '    except OSError: return saved(x)\n'
                '    return x * 0\n')
            protocol = {'deployment': {'bubblewrap': shutil.which('bwrap'), 'assay_python': sys.executable,
                        'control_device_node': '/dev/null', 'devices': {'0': {'render_node': '/dev/null'}}}}
            command = candidate_sandbox(root, SOURCE, bundle, protocol)
            command[-2] = 'cpu'
            with CandidateSession(command, environment=dict(os.environ), directory=evidence,
                                  stop=time.time() + 30, hard_stop=time.time() + 32,
                                  maximum_output_bytes=1024, device='cpu') as run:
                self.assertTrue(torch.equal(run(torch.ones(4)), torch.ones(4)))
            self.assertFalse(forbidden.exists())


class DeadlineFaultTests(unittest.TestCase):
    def test_deadline_guard_kills_hung_descendant_while_controller_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = '''import os, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pid = os.fork()
if pid == 0:
    while True:
        with open('heartbeats', 'a') as stream:
            stream.write(str(time.time()) + '\\n')
        time.sleep(0.01)
while True: time.sleep(60)
'''
            stop = time.time() + 0.4
            with patch('campaign.processes.parent_death_signal'), (patch.object(
                    ProcessIdentity, 'observe', side_effect=lambda pid: ProcessIdentity(pid, 'cpu-test'))
                    if sys.platform != 'linux' else contextlib.nullcontext()):
                with self.assertRaises(TimeoutError):
                    run_bounded([sys.executable, '-c', code], cwd=root, env=dict(os.environ),
                                stop=stop, hard_stop=stop + 1, log=root / 'log',
                                process_record=root / 'process.json', service=lambda: time.sleep(0.7))
            heartbeats = [float(row) for row in (root / 'heartbeats').read_text().splitlines()]
            self.assertLessEqual(max(heartbeats), stop + 0.05)
            size = (root / 'heartbeats').stat().st_size
            time.sleep(0.1)
            self.assertEqual((root / 'heartbeats').stat().st_size, size)


class AllocationTests(unittest.TestCase):
    def test_dropped_successor_arm_cannot_redefine_the_denominator(self):
        from test_campaign import OwnerTests, write
        from campaign.report import report
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = OwnerTests().allocation(root)
            document = json.loads(path.read_text())
            document['runs'] = [row for row in document['runs'] if row['condition'] == 'control']
            write(path, document)
            with self.assertRaisesRegex(ValueError, 'missing=.*b.*d'):
                reconcile_allocation(document)
            with self.assertRaisesRegex(ValueError, 'missing='):
                report(path)

    def test_changed_routing_or_replicate_is_refused(self):
        from test_campaign import OwnerTests
        with tempfile.TemporaryDirectory() as directory:
            path = OwnerTests().allocation(Path(directory))
            original = json.loads(path.read_text())
            for key, value in [('replicate', 2), ('physical_device', 7), ('compiler_commit', 'foreign')]:
                document = json.loads(json.dumps(original))
                document['runs'][0][key] = value
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'differs from frozen plan/routing'):
                    reconcile_allocation(document)


def campaign_coverage():
    from campaign.assay import isolated_measurement_coverage
    return isolated_measurement_coverage()
