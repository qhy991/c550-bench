"""Private frozen reference preparation and original-case qualification."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
import time

from .binding import clean_checkout, read_object, tracked_file
from .owner import publish_new


@contextmanager
def problem_root(root: Path):
    import c550bench
    previous = c550bench.ROOT
    c550bench.ROOT = root
    try:
        c550bench.upstream()
        yield c550bench
    finally:
        c550bench.ROOT = previous


def source_binding(root: Path) -> dict:
    binding = read_object(root / 'campaign/binding.json')
    return {key: binding[key] for key in
            ('task', 'baseline', 'bench_commit', 'compiler_commit', 'workload_ids')}


def prepare_reference(root: Path, source: Path, output: Path) -> dict:
    """Create the private selected adapter before device admission or author work."""
    selected = source_binding(root)
    clean_checkout(source, selected['bench_commit'])
    adapter = tracked_file(source, selected['baseline'], 'fixed reference')
    with problem_root(root) as bench:
        task = bench.task_record(selected['task'])
        _, workloads, _, _ = bench.load_problem(task)
        if [row.uuid for row in workloads] != selected['workload_ids']:
            raise ValueError('fixed reference workload order differs from the frozen original UUIDs')
        sources = bench.document('sources.lock.json')
    compile(adapter.read_text(), str(adapter), 'exec')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(adapter, output / 'adapter.py')
    (output / 'adapter.py').chmod(0o444)
    from .assay import prepare_maca_helper
    prepare_maca_helper(root, output / 'runtime-helper')
    prepared = {**selected, 'sources': sources, 'status': 'source_prepared',
                'device_setup': 'lazy backend compilation and device dispatch remain unqualified',
                'runtime_helper': str(output / 'runtime-helper'), 'performance': 'coverage_missing'}
    publish_new(output / 'source.json', prepared)
    return prepared


def compiled_original(original, route: dict, evidence: Path, *, workload_ids: tuple[str, ...]):
    """Retain emitted graphs and refuse compiler errors and recompile-limit fallback."""
    import torch
    import torch._dynamo.config as config
    from torch._dynamo.backends.registry import lookup_backend
    from torch._dynamo.utils import counters
    if route['capture'] not in ('fullgraph', 'declared_partial'):
        raise ValueError('unknown Inductor capture policy')
    if route['capture'] == 'declared_partial' and not route['known_eager_regions']:
        raise ValueError('partial compilation must declare its eager regions')
    # The captured MetaX convert_frame raises FailOnRecompileLimitHit for this flag.
    policy = {'suppress_errors': False, 'fail_on_recompile_limit_hit': True,
              'recompile_limit': len(workload_ids)}
    for name in policy:
        getattr(config, name)
    accumulated_limit = config.accumulated_recompile_limit
    backend = lookup_backend('inductor')
    import torch._inductor.config as inductor_config
    graphs = []
    before = dict(counters['graph_break'])

    def lower(graph, inputs, **options):
        index = len(graphs)
        (evidence / f'graph-{index:04d}.py').write_text(graph.code)
        graphs.append({'graph': index, 'inputs': [
            {'shape': list(value.shape), 'dtype': str(value.dtype), 'device': str(value.device)}
            if isinstance(value, torch.Tensor) else {'type': type(value).__name__}
            for value in inputs]})
        return backend(graph, inputs, **options)

    with config.patch(**policy):
        compiled = torch.compile(original.run, backend=lower,
                                 fullgraph=route['capture'] == 'fullgraph', dynamic=route['dynamic'])

    def run(*inputs):
        nonlocal accumulated_limit
        with config.patch(**policy):
            accumulated_limit = config.accumulated_recompile_limit
            result = compiled(*inputs)
        if route['capture'] == 'fullgraph' and not graphs:
            raise RuntimeError('strict reference produced no compiled graph')
        return result

    def diagnostics():
        return {'graphs': graphs, 'graph_breaks': {
            reason: count - before.get(reason, 0) for reason, count in counters['graph_break'].items()
            if count > before.get(reason, 0)}, 'fallback_policy': policy,
            'accumulated_recompile_limit': accumulated_limit,
            'inductor_compile_threads': inductor_config.compile_threads,
            'uniform_compilation_claim': False,
            'compiled_graphs_are_not_device_dispatch_or_performance_proof': True}
    return run, diagnostics


def qualify_reference(root: Path, request: dict, output: Path) -> dict:
    """Run only inside the existing bounded trusted assay, after canonical admission."""
    bundle = Path(request['reference_bundle'])
    selected = read_object(bundle / 'source.json')
    if any(selected.get(key) != value for key, value in source_binding(root).items()):
        raise ValueError('private reference source differs from the frozen selection')
    from .assay import admit_device, isolated_measurement_coverage
    admit_device(root, request, output, runtime_helper=Path(selected['runtime_helper']))
    import torch
    started = time.time()
    diagnostics = lambda: {}
    with problem_root(root) as bench:
        if selected['sources'] != bench.document('sources.lock.json'):
            raise ValueError('private reference sources differ from the original source lock')
        task = bench.task_record(selected['task'])
        _, _, _, path = bench.load_problem(task)
        original = bench.load_module(path / 'reference.py', 'c550bench_fixed_original')
        adapter = bench.load_module(bundle / 'adapter.py', 'c550bench_fixed_adapter')
        route = getattr(adapter, 'ROUTE', {'kind': 'tracked_callable', 'capture': 'not_declared'})
        publish_new(output / 'route.json', {'route': route, 'torch': torch.__version__,
                    'torch_source': torch.version.git_version, 'maca': getattr(torch.version, 'maca', None),
                    'float32_matmul_precision': torch.get_float32_matmul_precision(),
                    'allow_tf32': torch.backends.cuda.matmul.allow_tf32})
        if route['kind'] == 'inductor':
            call, diagnostics = compiled_original(
                original, route, output, workload_ids=tuple(selected['workload_ids']))
        elif route['kind'] == 'tracked_callable':
            call = adapter.run
        else:
            raise ValueError('unknown fixed reference route')
        try:
            correctness = bench.check_problem(
                task, device='cuda:0', candidate_path=bundle / 'adapter.py', symbol='run',
                reference_selfcheck=False, workload_scope='all',
                rounds=bench.document('suite.json')['correctness_rounds'],
                output=output / 'correctness.json', seed=bench.document('suite.json')['seed'],
                threads=4, candidate_callable=call)
        finally:
            publish_new(output / 'compilation.json', diagnostics())
    completed = time.time()
    if completed >= request['stop_at_epoch']:
        raise TimeoutError('fixed reference qualification completed after its original deadline')
    result = {'source': selected, 'route': route,
              'status': 'correct' if correctness['full_device_correctness'] else 'failed',
              'correctness': {key: correctness[key] for key in
                              ('status', 'full_device_correctness', 'selected_workloads', 'rounds')},
              'started_at_epoch': started, 'completed_at_epoch': completed,
              'measurement': isolated_measurement_coverage(), 'profile': isolated_measurement_coverage(),
              'performance': 'coverage_missing'}
    publish_new(output / 'result.json', result)
    return result


def qualified_reference(root: Path, receipt: Path) -> dict:
    result = read_object(receipt)
    if any(result.get('source', {}).get(key) != value for key, value in source_binding(root).items()):
        raise ValueError('reference qualification differs from the frozen selection')
    if result.get('status') != 'correct' or result.get('correctness', {}).get('full_device_correctness') is not True:
        raise ValueError('fixed reference lacks complete original device correctness')
    return result
