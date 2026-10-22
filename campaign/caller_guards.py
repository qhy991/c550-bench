"""Original-oracle caller checks; separate from original 16×10 numeric gate."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import c550bench as bench
from sol_execbench.core.bench.correctness import set_seed
from sol_execbench.core.bench.io import gen_inputs


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(snapshot(v) for v in value)
    if isinstance(value, list):
        return [snapshot(v) for v in value]
    return value


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()

    binding = json.loads((ROOT / 'campaign/binding.json').read_text())
    task_id = binding['task']

    bench.upstream()
    task_record = bench.task_record(task_id)
    definition, workloads, _, path = bench.load_problem(task_record)

    smoke_uuid = task_record['smoke_workload_uuid']
    w = next(x for x in workloads if x.uuid == smoke_uuid)

    custom = (
        getattr(bench.load_module(path / 'reference.py', 'guard_reference'),
                definition.custom_inputs_entrypoint)
        if definition.custom_inputs_entrypoint else None
    )
    ref = bench.load_module(path / 'reference.py', 'guard_reference')
    fn = bench.load_module(a.candidate, 'guard_candidate').run
    torch.set_num_threads(4)

    checks = []
    report = {
        'status': 'running',
        'task': task_id,
        'workload_uuid': w.uuid,
        'candidate_source_sha256': hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
        'scope': 'one predeclared smoke workload; caller checks do not replace original 16×10',
        'checks': checks,
    }

    def save():
        tmp = a.output.with_suffix(a.output.suffix + '.tmp')
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(a.output)

    with a.output.open('x') as f:
        json.dump(report, f)

    def compare(name, inputs):
        cloned = bench.cloned_inputs(inputs)
        expected = ref.run(*cloned)
        del cloned
        before = snapshot(inputs)
        actual = fn(*inputs)
        torch.cuda.synchronize()
        result = bench.compare_outputs(actual, expected, definition, w.tolerance, w.axes)
        result.update(name=name, inputs_unmutated=equal(inputs, before))
        checks.append(result)
        report['status'] = 'running' if (result['passed'] and result['inputs_unmutated']) else 'failed'
        save()
        assert result['passed'] and result['inputs_unmutated'], name
        del expected, before
        return actual

    set_seed(811)
    inputs = gen_inputs(definition, w, 'cuda:0', custom_inputs_fn=custom)
    first = compare('fresh_call', inputs)
    retained = snapshot(first)

    set_seed(812)
    fresh = gen_inputs(definition, w, 'cuda:0', custom_inputs_fn=custom)
    with torch.no_grad():
        for old, new in zip(inputs, fresh):
            if isinstance(old, torch.Tensor):
                old.set_(new)
            else:
                assert old == new, 'non-tensor input changed'
    second = compare('same_tensor_object_new_storage', inputs)
    del second, fresh

    retained_ok = equal(first, retained)
    checks.append({'name': 'retained_output_unchanged', 'passed': retained_ok})
    report['status'] = 'running' if retained_ok else 'failed'
    save()
    assert retained_ok, 'earlier output changed after later call'

    views = [v.view_as(v) if isinstance(v, torch.Tensor) else v for v in inputs]
    third = compare('fresh_view_same_storage', views)
    del third, views

    report['status'] = 'passed'
    save()
    print('caller guards passed', task_id, w.uuid)


if __name__ == '__main__':
    main()
