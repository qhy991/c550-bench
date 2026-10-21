"""Paired complete-callable wall-clock timing for engineering version comparison.

Both baseline and candidate run in the same admitted process (no isolation sandbox).
This is suitable for engineering version comparison only — not heldout attribution.

Each original workload receives three sets of 30 timed calls:
  forward:  baseline then candidate, alternating
  reverse:  candidate then baseline, alternating
  A/A:      baseline then baseline (control)

Speedup is conservative: min(baseline/candidate) over forward/reverse per workload,
geometric-meaned across all workloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def measure(fn, inputs: list, samples: int) -> list[float]:
    import torch
    values = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn(*inputs)
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1e6)
    return values


def conservative_geomean(rows: list[dict]) -> float:
    """Geometric mean of per-workload min-direction speedup ratios."""
    ratios = []
    for row in rows:
        b_fwd = row['baseline_us']['forward_median']
        b_rev = row['baseline_us']['reverse_median']
        c_fwd = row['candidate_us']['forward_median']
        c_rev = row['candidate_us']['reverse_median']
        ratios.append(min(b_fwd / c_fwd, b_rev / c_rev))
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios))


def main():
    import torch
    import c550bench
    from sol_execbench.core.bench.correctness import set_seed
    from sol_execbench.core.bench.io import gen_inputs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--gate', type=Path,
                        help='correctness result JSON; required unless --aa-mode')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--aa-mode', action='store_true',
                        help='A/A qualification: baseline==candidate, gate check skipped')
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f'output already exists: {args.output}')

    # Gate validation
    if args.aa_mode:
        if args.gate is not None:
            raise ValueError('--gate must not be provided in --aa-mode')
    else:
        if args.gate is None:
            raise ValueError('--gate is required unless --aa-mode is set')
        gate = json.loads(args.gate.read_text())
        if not (gate.get('full_device_correctness') and gate.get('status') == 'passed'
                and gate.get('task') == args.task):
            raise ValueError(
                'correctness gate requires full_device_correctness=True, status=passed, '
                f'and task={args.task!r}')
        gate_candidate = gate.get('candidate') or {}
        gate_path = gate_candidate.get('path')
        if gate_path is None or Path(gate_path).resolve() != args.candidate.resolve():
            raise ValueError('gate candidate path does not match --candidate')

    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

    c550bench.upstream()
    task = c550bench.task_record(args.task)
    definition, workloads, _, folder = c550bench.load_problem(task)
    reference_module = c550bench.load_module(folder / 'reference.py', 'c550bench_reference')
    custom_name = definition.custom_inputs_entrypoint
    custom = getattr(reference_module, custom_name) if custom_name else None

    baseline_fn = c550bench.load_module(args.baseline, 'c550bench_baseline').run
    candidate_fn = c550bench.load_module(args.candidate, 'c550bench_candidate').run

    torch.set_num_threads(4)
    rows = []
    for workload in workloads:
        set_seed(200)
        inputs = gen_inputs(definition, workload, args.device, custom_inputs_fn=custom)
        snapshots = [v.clone() if isinstance(v, torch.Tensor) else v for v in inputs]

        # warmup: 10 pairs (one baseline call + one candidate call each)
        for _ in range(10):
            baseline_fn(*inputs)
            candidate_fn(*inputs)

        streams: dict[str, list[float]] = {k: [] for k in ('bf', 'cf', 'br', 'cr', 'a1', 'a2')}
        for _ in range(args.samples):
            streams['bf'] += measure(baseline_fn, inputs, 1)
            streams['cf'] += measure(candidate_fn, inputs, 1)
        for _ in range(args.samples):
            streams['cr'] += measure(candidate_fn, inputs, 1)
            streams['br'] += measure(baseline_fn, inputs, 1)
        for _ in range(args.samples):
            streams['a1'] += measure(baseline_fn, inputs, 1)
            streams['a2'] += measure(baseline_fn, inputs, 1)

        med = {k: statistics.median(v) for k, v in streams.items()}
        unchanged = all(
            torch.equal(v, s) if isinstance(v, torch.Tensor) else v == s
            for v, s in zip(inputs, snapshots)
        )
        row = {
            'workload_uuid': workload.uuid,
            'axes': workload.axes,
            'inputs_unmutated_after_measurement': unchanged,
            'baseline_us': {
                'median': statistics.median(streams['bf'] + streams['br']),
                'forward_median': med['bf'],
                'reverse_median': med['br'],
            },
            'candidate_us': {
                'median': statistics.median(streams['cf'] + streams['cr']),
                'forward_median': med['cf'],
                'reverse_median': med['cr'],
            },
            'aa_control_baseline_us': {'abs_drift': abs(med['a1'] - med['a2'])},
            'samples_us': streams,
        }
        rows.append(row)
        print(workload.uuid, row['baseline_us']['median'], row['candidate_us']['median'], flush=True)
        del inputs, snapshots
        torch.cuda.empty_cache()

    result = {
        'task': args.task,
        'method': 'paired_complete_callable_wall',
        'scope': ('engineering version comparison; candidate and baseline run in the same process; '
                  'not suitable for heldout or independent clean-start attribution'),
        'device': args.device,
        'seed': 200,
        'warmup_calls': 10,
        'samples_per_direction': args.samples,
        'aa_mode': args.aa_mode,
        'candidate_source_sha256': sha(args.candidate),
        'baseline_source_sha256': sha(args.baseline),
        'conservative_geomean': conservative_geomean(rows),
        'rows': rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')

    geomean = result['conservative_geomean']
    print(f'conservative_geomean={geomean:.4f}  rows={len(rows)}', flush=True)


if __name__ == '__main__':
    main()
