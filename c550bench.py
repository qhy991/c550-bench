#!/usr/bin/env python3
"""A C550 correctness-only suite reusing pinned SOL-ExecBench data and helpers."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent


def document(name):
    return json.loads((ROOT / name).read_text())


def upstream():
    if sys.version_info < (3, 10):
        raise RuntimeError("audit/check require Python >=3.10 in the existing MetaX environment; list uses only stdlib")
    source = ROOT / ".deps/sol-execbench"
    expected = document("sources.lock.json")["sol_execbench"]["revision"]
    git = ["git", "-c", "safe.directory=" + str(source), "-C", str(source)]
    observed = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    if observed != expected:
        raise RuntimeError("SOL-ExecBench revision differs from sources.lock.json")
    subprocess.run(git + ["diff", "--quiet", "HEAD", "--", "src/sol_execbench/core"], check=True)
    sys.path.insert(0, str(source / "src"))
    return source


def task_record(task_id):
    for task in document("suite.json")["tasks"]:
        if task["id"] == task_id:
            return task
    raise ValueError("Task is not in this suite: " + task_id)


def load_problem(task):
    from sol_execbench.core.data import Definition, Workload
    path = ROOT / ".data/benchmark" / task["id"]
    raw = json.loads((path / "definition.json").read_text())
    if (path / "reference.py").read_text() != raw["reference"]:
        raise RuntimeError("reference.py projection differs from definition.json")
    definition = Definition.model_validate(raw)
    raw_workloads = [json.loads(line) for line in (path / "workload.jsonl").read_text().splitlines() if line.strip()]
    workloads = [Workload.model_validate(w) for w in raw_workloads]
    if len({w.uuid for w in workloads}) != len(workloads):
        raise ValueError("Duplicate workload UUID")
    return definition, workloads, raw_workloads, path


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_suite():
    upstream()
    dtype_size = {"bfloat16": 2, "float16": 2, "float32": 4, "float64": 8, "int64": 8, "int32": 4, "int16": 2, "int8": 1, "bool": 1}
    rows = []
    for task in document("suite.json")["tasks"]:
        definition, workloads, raw_workloads, _ = load_problem(task)
        cases = []
        for workload, raw in zip(workloads, raw_workloads):
            shapes = definition.get_input_shapes(workload.axes)
            byte_count = sum(math.prod(s) * dtype_size[definition.inputs[n].dtype.value] for n, s in shapes.items() if s is not None)
            cases.append({"uuid": workload.uuid, "axes": workload.axes, "input_bytes": byte_count,
                          "raw_tolerance": raw.get("tolerance", {}),
                          "effective_tolerance": workload.tolerance.model_dump(),
                          "unrecognized_tolerance_fields": sorted(set(raw.get("tolerance", {})) - set(type(workload.tolerance).model_fields))})
        smoke = next(c for c in cases if c["uuid"] == task["smoke_workload_uuid"])
        rows.append({"task": task["id"], "workload_count": len(cases), "smoke": smoke,
                     "custom_inputs_entrypoint": definition.custom_inputs_entrypoint,
                     "min_input_bytes": min(c["input_bytes"] for c in cases),
                     "max_input_bytes": max(c["input_bytes"] for c in cases), "cases": cases})
    return {"sources": document("sources.lock.json"), "tasks": rows,
            "total_workloads": sum(r["workload_count"] for r in rows)}


def cloned_inputs(inputs):
    import torch
    return [value.detach().clone(memory_format=torch.preserve_format) if isinstance(value, torch.Tensor) else value for value in inputs]


def compare_outputs(candidate, reference, definition, tolerance, axes):
    """Retain upstream numeric rules; additionally require exact discrete outputs."""
    import torch
    from sol_execbench.core.bench.correctness import compute_error_stats
    from sol_execbench.core.bench.io import normalize_outputs
    from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype
    names = list(definition.outputs)
    dtypes = {n: dtype_str_to_torch_dtype(s.dtype) for n, s in definition.outputs.items()}
    for obj in (reference, candidate):
        if isinstance(obj, dict) and set(obj) != set(names):
            return {"passed": False, "reason": "output_names", "outputs": {}}
    # All selected references return tensors, tensor tuples, or dictionaries.
    first = reference if isinstance(reference, torch.Tensor) else next(iter(reference.values())) if isinstance(reference, dict) else reference[0]
    device = first.device
    ref = normalize_outputs(reference, device=device, output_names=names, output_dtypes=dtypes)
    out = normalize_outputs(candidate, device=device, output_names=names, output_dtypes=dtypes)
    shapes = definition.get_output_shapes(axes)
    checks = {}
    for name in names:
        actual, expected = out[name], ref[name]
        if expected.dtype != dtypes[name] or tuple(expected.shape) != tuple(shapes[name] or ()):
            checks[name] = {"passed": False, "reason": "invalid_reference_contract"}
        elif actual.shape != expected.shape:
            checks[name] = {"passed": False, "reason": "shape"}
        elif actual.dtype != expected.dtype:
            checks[name] = {"passed": False, "reason": "dtype"}
        elif not expected.is_floating_point():
            # Float32 conversion plus relative tolerance can accept wrong token indices.
            equal = torch.equal(actual, expected)
            checks[name] = {"passed": equal, "reason": "exact_discrete", "mismatches": int((actual != expected).sum().item())}
        else:
            metrics, exceeds = compute_error_stats(actual, expected, tolerance)
            checks[name] = {"passed": not exceeds, "reason": "upstream_numeric", **metrics.model_dump()}
    return {"passed": all(v["passed"] for v in checks.values()), "outputs": checks}


def validate_c550_identity(device_name, maca_version, expected_name):
    """The MetaX CUDA compatibility capability is not the physical xcore target."""
    if not maca_version or device_name != expected_name:
        raise RuntimeError("Expected MetaX C550 runtime/device, got " + str(device_name))


def check_problem(task, *, device, candidate_path, symbol, reference_selfcheck, workload_scope, rounds, output, seed, threads):
    import torch
    from sol_execbench.core.bench.correctness import set_seed
    from sol_execbench.core.bench.io import gen_inputs
    definition, workloads, raw_workloads, path = load_problem(task)
    if workload_scope != "all":
        chosen = task["smoke_workload_uuid"] if workload_scope == "smoke" else workload_scope
        workloads = [w for w in workloads if w.uuid == chosen]
        if not workloads:
            raise ValueError("Unknown workload UUID: " + chosen)
    reference = load_module(path / "reference.py", "c550bench_reference")
    custom = getattr(reference, definition.custom_inputs_entrypoint) if definition.custom_inputs_entrypoint else None
    if any(getattr(v, "type", "") == "safetensors" for w in workloads for v in w.inputs.values()):
        raise ValueError("This suite requires complete local input sources; safetensors is not enabled")
    candidate = reference.run if reference_selfcheck else getattr(load_module(candidate_path, "c550bench_candidate"), symbol)
    if torch.device(device).type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA-compatible MetaX device is unavailable")
        props = torch.cuda.get_device_properties(torch.device(device))
        maca_version = getattr(torch.version, "maca", None)
        validate_c550_identity(props.name, maca_version, document("suite.json")["target_device_name"])
        arch = document("suite.json")["target_arch"]
        device_name = props.name
    elif torch.device(device).type == "cpu":
        arch = "cpu"
        device_name = "CPU"
        maca_version = getattr(torch.version, "maca", None)
    else:
        raise ValueError("Device must be cpu or the MetaX CUDA-compatible device")
    torch.set_num_threads(threads)
    set_seed(seed)
    report = {"task": task["id"], "sources": document("sources.lock.json"), "device": device, "arch": arch,
              "torch": torch.__version__, "maca": maca_version, "device_name": device_name, "mode": "reference_selfcheck" if reference_selfcheck else "candidate_correctness",
              "candidate": None if reference_selfcheck else {"path": str(candidate_path), "symbol": symbol},
              "seed": seed, "rounds": rounds, "workload_scope": workload_scope, "expected_workloads": len(raw_workloads),
              "selected_workloads": len(workloads), "expected_rounds": document("suite.json")["correctness_rounds"],
              "cases": [], "status": "running", "full_device_correctness": False, "performance": "not_measured"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as initial:
        json.dump(report, initial)
    def save():
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output)
    for workload in workloads:
        for round_index in range(rounds):
            stage = "input_generation"
            row = {"workload_uuid": workload.uuid, "round": round_index, "axes": workload.axes,
                   "tolerance": workload.tolerance.model_dump()}
            try:
                inputs = gen_inputs(definition, workload, device, custom_inputs_fn=custom)
                stage = "reference"
                reference_inputs = cloned_inputs(inputs)
                expected = reference.run(*reference_inputs)
                del reference_inputs
                if arch != "cpu":
                    torch.cuda.synchronize(torch.device(device))
                stage = "candidate"
                actual = candidate(*inputs)
                if arch != "cpu":
                    torch.cuda.synchronize(torch.device(device))
                stage = "comparison"
                row.update(compare_outputs(actual, expected, definition, workload.tolerance, workload.axes))
                del inputs, actual, expected
            except Exception as exc:
                row.update(passed=False, failure_stage=stage, error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc())
            report["cases"].append(row)
            if not row["passed"]:
                report["status"] = "failed"
                save()
                return report
            save()
    report["status"] = "passed"
    report["full_device_correctness"] = bool(not reference_selfcheck and arch == document("suite.json")["target_arch"] and workload_scope == "all" and rounds == report["expected_rounds"])
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    audit = commands.add_parser("audit")
    audit.add_argument("--output", type=Path)
    check = commands.add_parser("check")
    check.add_argument("--task", required=True)
    check.add_argument("--device", required=True, help="cpu or the admitted logical MetaX device, e.g. cuda:0")
    route = check.add_mutually_exclusive_group(required=True)
    route.add_argument("--candidate", type=Path, help="Python module exposing the original ordered return-value ABI")
    route.add_argument("--reference-selfcheck", action="store_true")
    check.add_argument("--symbol", default="run")
    check.add_argument("--workloads", default="all", help="all, smoke, or one original workload UUID")
    check.add_argument("--rounds", type=int, default=document("suite.json")["correctness_rounds"])
    check.add_argument("--seed", type=int, default=document("suite.json")["seed"])
    check.add_argument("--threads", type=int, default=4)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "list":
        for task in document("suite.json")["tasks"]:
            print("{id}\tdifficulty={difficulty}\t{label}".format(**task))
        return 0
    if args.command == "audit":
        result = audit_suite()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as out:
                json.dump(result, out, indent=2)
        print(json.dumps({"tasks": len(result["tasks"]), "workloads": result["total_workloads"]}))
        return 0
    if args.rounds < 1 or args.threads < 1:
        parser.error("rounds and threads must be positive")
    upstream()
    result = check_problem(task_record(args.task), device=args.device, candidate_path=args.candidate, symbol=args.symbol,
                           reference_selfcheck=args.reference_selfcheck, workload_scope=args.workloads, rounds=args.rounds,
                           output=args.output, seed=args.seed, threads=args.threads)
    print(json.dumps({k: result[k] for k in ("task", "status", "mode", "selected_workloads", "rounds", "full_device_correctness")}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
