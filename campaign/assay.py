"""Original whole-task correctness, paired full-call events and separate MCPTI."""
from __future__ import annotations

from dataclasses import asdict
import math
import platform
from importlib.metadata import version
from pathlib import Path
import statistics
import sys
import time

from .binding import read_object
from .owner import publish_new

TIMING = read_object(Path(__file__).with_name('protocol.json'))['timing']
TIMER = TIMING['timer']
INTERVAL = TIMING['measured_interval']
RESET = TIMING['device_state_reset']
WARMUPS = TIMING['sample_policy']['warmups']
PAIRS = TIMING['sample_policy']['paired_samples_per_workload']
MAX_RELATIVE_MAD = TIMING['sample_policy']['maximum_relative_mad']


def paired_summary(rows: list[dict]) -> dict:
    ratios = []
    quality = bool(rows)
    for row in rows:
        groups = {role: [pair[role] for pair in row['pairs']] for role in ('baseline', 'candidate')}
        if len(row['pairs']) != PAIRS or any(
                type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                for values in groups.values() for value in values):
            raise ValueError('whole-call timing requires ten finite positive paired samples')
        centers = {role: statistics.median(values) for role, values in groups.items()}
        spreads = {role: statistics.median(abs(value - centers[role]) for value in values)
                   / centers[role] for role, values in groups.items()}
        quality = quality and all(value <= MAX_RELATIVE_MAD for value in spreads.values())
        ratio = centers['candidate'] / centers['baseline']
        ratios.append(ratio)
        row.update(median_ms=centers, relative_mad=spreads, candidate_over_baseline=ratio)
    return {'timer': TIMER, 'interval': INTERVAL, 'reset': RESET, 'warmups': WARMUPS,
            'pairs_per_workload': PAIRS, 'paired_order': 'alternating_baseline_candidate',
            'quality_passed': quality,
            'candidate_over_baseline': math.exp(statistics.mean(math.log(ratio) for ratio in ratios)),
            'workloads': rows, 'profiler_enabled': False}


class WholeCallTimer:
    def __init__(self, torch, l2_bytes: int):
        self.torch = torch
        if type(l2_bytes) is not int or l2_bytes <= 0:
            raise ValueError('whole-call reset requires declared target L2')
        if torch.cuda.get_device_properties(0).L2_cache_size != l2_bytes:
            raise ValueError('observed L2 differs from target')
        self.stream = torch.cuda.default_stream(0)
        self.reset = torch.empty(TIMING['reset_l2_multiple'] * l2_bytes // 4,
                                 dtype=torch.float32, device='cuda:0')
        self.begin = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.stream):
            self.reset.fill_(1)
            self.begin.record(self.stream)
            self.end.record(self.stream)
            self.end.synchronize()

    def sample(self, function, inputs):
        import c550bench
        torch = self.torch
        values = c550bench.cloned_inputs(inputs)
        with torch.cuda.stream(self.stream):
            self.reset.fill_(1)
            torch.cuda.synchronize(0)
            self.begin.record(self.stream)
            self.begin.synchronize()
            output = function(*values)
            torch.cuda.synchronize(0)
            self.end.record(self.stream)
            self.end.synchronize()
            elapsed = float(self.begin.elapsed_time(self.end))
        del output, values
        return elapsed


def profile_full_call(torch, function, inputs, library: str) -> dict:
    import c550bench
    from open_cake_ir.evaluation.metax_activity import activity_collector
    collector = activity_collector(library)
    values = c550bench.cloned_inputs(inputs)
    torch.cuda.synchronize(0)
    collector.begin()
    try:
        output = function(*values)
        torch.cuda.synchronize(0)
    finally:
        activity = collector.finish()
    del output, values
    if (activity.get('dropped_records') != 0 or activity.get('pending_buffers') != 0
            or activity.get('collection_errors') or not activity.get('records')):
        raise ValueError('whole-call MCPTI coverage is incomplete')
    kernels = [row for row in activity['records'] if row['kind'] == 10]
    if not kernels or any(row.get('device') != 0 or row.get('end_ns', 0) <= row.get('start_ns', 0)
                          for row in kernels):
        raise ValueError('whole-call MCPTI has no complete admitted device activity')
    unmodeled = sorted({row['kind'] for row in activity['records']} - {4, 5, 10})
    return {'status': 'coverage_missing' if unmodeled else 'passed',
            'unmodeled_activity_kinds': unmodeled, 'kind': 'maca_complete_callable_activity',
            'kernel_count': len(kernels), 'activity': activity,
            'scope': 'complete invocation, separate from unprofiled event samples'}


def validate_runtime_capture(root: Path):
    host = read_object(root / '.deps/cake-ir/runtime/hosts/xcore1002.json')['host_environment']
    if platform.python_version() != host['python']['version']:
        raise ValueError('assay Python differs from the selected Compiler host capture')
    for package, expected in host['packages'].items():
        if version(package) != expected:
            raise ValueError(f'assay {package} differs from the selected Compiler host capture')


def evaluate(root: Path, request: dict, output: Path) -> dict:
    """Invoked only by the supervisor's isolated, bounded assay subprocess."""
    import c550bench
    validate_runtime_capture(root)
    binding = read_object(root / 'campaign/binding.json')
    protocol = read_object(root / binding['measurement_protocol'])
    deadline = request['stop_at_epoch']
    if time.time() >= deadline:
        raise TimeoutError('assay deadline closed before device admission')
    compiler = root / '.deps/cake-ir'
    sys.path.insert(0, str(compiler / 'src'))
    from open_cake_ir.evaluation.local_broker import admit_local_job
    from open_cake_ir.evaluation.triton_metax import observe_local_metax
    from open_cake_ir.compiler.target import declared_target
    route = protocol['deployment']['devices'][str(binding['physical_device'])]
    job = admit_local_job('maca', device=binding['physical_device'], lock_scope='device',
                          runtime_device=route['runtime_device'], expected_pci=route['expected_pci'],
                          queue_seconds=max(0, min(300, deadline - time.time())))
    runtime = str(Path(protocol['deployment']['runtime_library']).resolve(strict=True))
    admission = observe_local_metax('xcore1002', runtime_library=runtime)
    publish_new(output / 'admission.json', {'job': job, **asdict(admission)})
    import torch
    torch.set_num_threads(4)
    c550bench.ROOT = root
    c550bench.upstream()
    task = c550bench.task_record(binding['task'])
    candidate = Path(request['snapshot']) / 'candidate.py'
    sys.path.insert(0, str(candidate.parent))
    correctness = c550bench.check_problem(
        task, device='cuda:0', candidate_path=candidate, symbol='run',
        reference_selfcheck=False, workload_scope='all', rounds=10,
        output=output / 'correctness.json', seed=c550bench.document('suite.json')['seed'], threads=4)
    rows = {}
    for case in correctness['cases']:
        cell = rows.setdefault(case['workload_uuid'], {'correctness': {'status': 'passed', 'rounds': 0}})
        cell['correctness']['rounds'] += 1
        if not case['passed']:
            cell['correctness']['status'] = 'failed'
    result = {'id': request['id'], 'candidate': request['candidate'], 'phase': request['phase'],
              'status': 'incorrect', 'correctness': {'status': correctness['status'],
              'full_device_correctness': correctness['full_device_correctness']}, 'workloads': rows}
    if not correctness['full_device_correctness']:
        result['completed_at_epoch'] = time.time()
        publish_new(output / 'result.json', result)
        return result
    definition, workloads, _, task_root = c550bench.load_problem(task)
    if [row.uuid for row in workloads] != binding['workload_ids']:
        raise ValueError('assay workload order differs from frozen original UUIDs')
    reference = c550bench.load_module(task_root / 'reference.py', 'whole_task_reference')
    custom = getattr(reference, definition.custom_inputs_entrypoint) if definition.custom_inputs_entrypoint else None
    from sol_execbench.core.bench.io import gen_inputs
    from sol_execbench.core.bench.correctness import set_seed
    set_seed(c550bench.document('suite.json')['seed'])
    candidate_run = c550bench.load_module(candidate, 'whole_task_candidate').run
    baseline_run = c550bench.load_module(root / binding['baseline'], 'whole_task_baseline').run
    baseline = c550bench.check_problem(
        task, device='cuda:0', candidate_path=root / binding['baseline'], symbol='run',
        reference_selfcheck=False, workload_scope='all', rounds=10,
        output=output / 'baseline-correctness.json', seed=c550bench.document('suite.json')['seed'], threads=4)
    if not baseline['full_device_correctness']:
        raise ValueError('fixed baseline failed the original whole-task oracle')
    observations = []
    timer = WholeCallTimer(torch, declared_target('xcore1002').l2_cache_bytes)
    for workload in workloads:
        if time.time() >= deadline:
            raise TimeoutError('whole-task assay exceeded its original deadline')
        inputs = gen_inputs(definition, workload, 'cuda:0', custom_inputs_fn=custom)
        if request['phase'] == 'profile':
            profiles = {}
            for role, function in (('baseline', baseline_run), ('candidate', candidate_run)):
                for _ in range(WARMUPS):
                    values = c550bench.cloned_inputs(inputs)
                    value = function(*values)
                    torch.cuda.synchronize(0)
                    del values, value
                profiles[role] = profile_full_call(torch, function, inputs,
                    str(Path(protocol['deployment']['activity_library']).resolve(strict=True)))
            publish_new(output / f'profile-{workload.uuid}.json', profiles)
            rows[workload.uuid]['profile'] = {'status': 'passed' if all(
                value['status'] == 'passed' for value in profiles.values()) else 'coverage_missing'}
        else:
            for function in (baseline_run, candidate_run):
                for _ in range(WARMUPS):
                    timer.sample(function, inputs)
            pairs = []
            for index in range(PAIRS):
                pair = {}
                order = ('baseline', 'candidate') if index % 2 == 0 else ('candidate', 'baseline')
                for role in order:
                    if time.time() >= deadline:
                        raise TimeoutError('timing exceeded its original deadline')
                    pair[role] = timer.sample(baseline_run if role == 'baseline' else candidate_run, inputs)
                pairs.append(pair)
            observations.append({'uuid': workload.uuid, 'pairs': pairs})
            publish_new(output / f'timing-{workload.uuid}.json', observations[-1])
        del inputs
    result['status'] = 'passed'
    if request['phase'] != 'profile':
        result['measurement'] = paired_summary(observations)
        for row in observations:
            rows[row['uuid']]['measurement'] = {'status': 'passed' if result['measurement']['quality_passed'] else 'quality_failed', **row}
    else:
        complete = all(row['profile']['status'] == 'passed' for row in rows.values())
        result['status'] = 'passed' if complete else 'coverage_missing'
        result['profile'] = {'status': result['status'], 'workloads': len(workloads)}
    result['completed_at_epoch'] = time.time()
    if result['completed_at_epoch'] >= deadline:
        raise TimeoutError('assay completed after its original deadline')
    publish_new(output / 'result.json', result)
    return result
