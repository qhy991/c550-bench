"""Trusted whole-task oracle with isolated candidates and explicit measurement coverage."""
from __future__ import annotations

from dataclasses import asdict
import math
import os
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


def validate_runtime_capture(root: Path):
    host = read_object(root / '.deps/cake-ir/runtime/hosts/xcore1002.json')['host_environment']
    if platform.python_version() != host['python']['version']:
        raise ValueError('assay Python differs from the selected Compiler host capture')
    for package, expected in host['packages'].items():
        if version(package) != expected:
            raise ValueError(f'assay {package} differs from the selected Compiler host capture')


def admit_device(root: Path, request: dict, output: Path) -> None:
    """Shared admission for the reference opportunity and isolated candidate assay."""
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
    route = protocol['deployment']['devices'][str(binding['physical_device'])]
    job = admit_local_job('maca', device=binding['physical_device'], lock_scope='device',
                          runtime_device=route['runtime_device'], expected_pci=route['expected_pci'],
                          queue_seconds=max(0, min(300, deadline - time.time())))
    runtime = str(Path(protocol['deployment']['runtime_library']).resolve(strict=True))
    admission = observe_local_metax('xcore1002', runtime_library=runtime)
    publish_new(output / 'admission.json', {'job': job, **asdict(admission)})
    import torch
    torch.set_num_threads(4)


def evaluate(root: Path, request: dict, output: Path) -> dict:
    """Invoked only by the supervisor's isolated, bounded assay subprocess."""
    import c550bench
    from .reference import qualified_reference
    baseline = qualified_reference(root, Path(request['baseline_receipt']))
    admit_device(root, request, output)
    binding = read_object(root / 'campaign/binding.json')
    protocol = read_object(root / binding['measurement_protocol'])
    deadline = request['stop_at_epoch']
    import torch
    c550bench.ROOT = root
    c550bench.upstream()
    task = c550bench.task_record(binding['task'])
    definition, workloads, _, _ = c550bench.load_problem(task)
    if [row.uuid for row in workloads] != binding['workload_ids']:
        raise ValueError('assay workload order differs from frozen original UUIDs')
    from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype
    maximum_output_bytes = max(sum(
        math.prod(shape or ()) * torch.empty((), dtype=dtype_str_to_torch_dtype(
            definition.outputs[name].dtype)).element_size()
        for name, shape in definition.get_output_shapes(workload.axes).items())
        for workload in workloads)
    from .isolated import CandidateSession
    from .processes import candidate_sandbox, runtime_environment
    candidate = Path(request['snapshot'])
    worker_output = output / 'worker'
    worker_output.mkdir()
    # The untrusted process receives only input values and returns value bytes.
    # The canonical oracle and result files never enter its namespace or descriptors.
    environment = runtime_environment(root)
    for key in ('MACA_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'MC_VISIBLE_DEVICES'):
        if key in os.environ:
            environment[key] = os.environ[key]
    with CandidateSession(candidate_sandbox(root, Path(__file__).resolve().parents[1],
                          candidate, protocol), environment=environment, directory=worker_output,
                          stop=deadline, hard_stop=request['hard_stop_at_epoch'],
                          maximum_output_bytes=maximum_output_bytes, device='cuda:0') as run:
        correctness = c550bench.check_problem(
            task, device='cuda:0', candidate_path=candidate / 'candidate.py', symbol='run',
            reference_selfcheck=False, workload_scope='all', rounds=10,
            output=output / 'correctness.json', seed=c550bench.document('suite.json')['seed'],
            threads=4, candidate_callable=run)
        if run.failure is not None:
            raise RuntimeError('candidate transport outcome unknown; no retry: ' + run.failure)
    rows = {}
    for case in correctness['cases']:
        cell = rows.setdefault(case['workload_uuid'], {'correctness': {'status': 'passed', 'rounds': 0}})
        cell['correctness']['rounds'] += 1
        if not case['passed']:
            cell['correctness']['status'] = 'failed'
    result = {'id': request['id'], 'candidate': request['candidate'], 'phase': request['phase'],
              'status': 'coverage_missing' if correctness['full_device_correctness'] else 'incorrect',
              'correctness': {'status': correctness['status'],
                              'full_device_correctness': correctness['full_device_correctness']},
              'workloads': rows, 'baseline': baseline, 'measurement': isolated_measurement_coverage(),
              'profile': isolated_measurement_coverage(), 'completed_at_epoch': time.time()}
    if result['completed_at_epoch'] >= deadline:
        raise TimeoutError('assay completed after its original deadline')
    publish_new(output / 'result.json', result)
    return result


def isolated_measurement_coverage() -> dict:
    return {'status': 'coverage_missing', 'quality_passed': False,
            'reason': 'no qualified trusted cross-process complete-call timer or MCPTI collector',
            'required_evidence': 'C550 worker-context completion, complete callable interval, reset, '
                                 'paired baseline samples and independent MCPTI ownership'}
