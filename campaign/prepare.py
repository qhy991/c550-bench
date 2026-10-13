"""Create fresh whole-task campaign workspaces without provider or GPU work."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .binding import (
    CONDITIONS, clean_checkout, git, host_config, read_object, validate_plan,
)


GENERATED = (
    "campaign/candidates/", "campaign/schedules/", "campaign/compiler-actions/",
    "campaign/evaluations/", "campaign/final-confirmation/", "campaign/management/",
    "campaign/deadline.json", "campaign/intake.json", "campaign/ENDPOINT.json",
    "campaign/DONE.json", "campaign/JOURNAL.md",
)


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def source_identity(root: Path) -> dict[str, str]:
    identity = {}
    for key in ("user.name", "user.email"):
        value = git(root, "config", "--get", key)
        if not value:
            raise ValueError(f"configure Bench checkout Git {key} before preparation")
        identity[key] = value
    return identity


def clone(source: Path, destination: Path, commit: str) -> None:
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(source), str(destination)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--quiet", "--detach", commit],
        check=True,
    )


def _host_inputs(plan: dict, config: dict) -> tuple[Path, Path, dict[str, Path], Path, str]:
    required = {"bench_root", "data_root", "compiler_roots", "hmz_adapter",
                "devices", "start_window_seconds"}
    if set(config) != required:
        raise ValueError("host configuration fields differ")
    bench_root = clean_checkout(Path(config["bench_root"]), plan["bindings"]["bench"]["commit"])
    data_root = Path(config["data_root"]).resolve(strict=True)
    if not (data_root / "benchmark").is_dir():
        raise ValueError("data_root must contain benchmark task material")
    compiler_roots = {}
    if set(config["compiler_roots"]) != set(CONDITIONS):
        raise ValueError("host requires control and successor Compiler roots")
    for condition in CONDITIONS:
        compiler_roots[condition] = clean_checkout(
            Path(config["compiler_roots"][condition]),
            plan["bindings"][condition]["commit"],
        )
    adapter = config["hmz_adapter"]
    if not isinstance(adapter, dict) or set(adapter) != {"path", "source_commit"}:
        raise ValueError("host requires the pinned shared hmz adapter")
    adapter_path = Path(adapter["path"]).resolve(strict=True)
    if not adapter_path.is_file() or not isinstance(adapter["source_commit"], str):
        raise ValueError("hmz adapter binding is invalid")
    adapter_root = Path(git(adapter_path.parent, "rev-parse", "--show-toplevel"))
    relative = adapter_path.relative_to(adapter_root)
    frozen_bytes = subprocess.check_output(
        ["git", "-C", str(adapter_root), "show", f"{adapter['source_commit']}:{relative}"],
        stderr=subprocess.PIPE,
    )
    if frozen_bytes != adapter_path.read_bytes():
        raise ValueError("hmz adapter differs from its frozen source commit")
    return bench_root, data_root, compiler_roots, adapter_path, adapter["source_commit"]


def prepare(plan_path: Path, routing_path: Path, host: str, output: Path) -> dict:
    plan, routing = read_object(plan_path), read_object(routing_path)
    output = Path(output).absolute()
    if output != output.resolve() or any((parent / ".git").exists() for parent in output.parents):
        raise ValueError("allocation output must be canonical and outside Git checkouts")
    if output.exists():
        raise FileExistsError(output)
    config, selected = host_config(plan, routing, host)
    bench_root, data_root, compiler_roots, adapter_path, adapter_commit = _host_inputs(plan, config)
    workload_ids = validate_plan(plan, bench_root, data_root)
    identity = source_identity(bench_root)
    devices = {row["id"]: row for row in config["devices"]}
    assignment_ids = [Path(row["workspace"]).name for row in selected]
    if (len(set(assignment_ids)) != len(assignment_ids)
            or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value)
                   for value in assignment_ids)):
        raise ValueError("frozen workspace names must be unique simple assignment identities")
    records = []
    for assignment in selected:
        run = Path(assignment["root"])
        if run.exists():
            raise ValueError(f"fresh campaign workspace already exists: {run}")
        run.parent.mkdir(parents=True, exist_ok=True)
        clone(bench_root, run, plan["bindings"]["bench"]["commit"])
        for key, value in identity.items():
            subprocess.run(["git", "-C", str(run), "config", "--local", key, value], check=True)
        (run / ".deps").mkdir()
        compiler_commit = plan["bindings"][assignment["condition"]]["commit"]
        clone(compiler_roots[assignment["condition"]], run / ".deps/cake-ir", compiler_commit)
        for name in ("candidates", "schedules", "compiler-actions", "evaluations",
                     "final-confirmation", "management"):
            (run / "campaign" / name).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(adapter_path, run / "campaign/hmz_compiler_tools.py")
        author = dict(plan["author"])
        binding = {
            "schema_version": 1,
            "kind": "c550_bench_whole_task",
            "assignment_id": Path(assignment["workspace"]).name,
            "frozen_workspace": assignment["workspace"],
            "task": assignment["task"],
            "condition": assignment["condition"],
            "replicate": assignment["replicate"],
            "root": str(run),
            "host": host,
            "device_id": assignment["device_id"],
            "physical_device": devices[assignment["device_id"]]["physical_device"],
            "author_capacity": devices[assignment["device_id"]]["author_capacity"],
            "bench_commit": plan["bindings"]["bench"]["commit"],
            "compiler_commit": compiler_commit,
            "hmz_adapter_source_commit": adapter_commit,
            "author": author,
            "measurement_protocol": plan["measurement_protocol"],
            "baseline": plan["baseline_files"][assignment["task"]],
            "workload_ids": list(workload_ids[assignment["task"]]),
            "correctness_rounds": read_object(bench_root / "suite.json")["correctness_rounds"],
        }
        write_new(run / "campaign/frozen-plan.json", plan)
        write_new(run / "campaign/frozen-routing.json", routing)
        write_new(run / "campaign/binding.json", binding)
        write_new(run / "campaign/data-binding.json", {
            "root": str(data_root),
            "task_root": str(data_root / "benchmark" / assignment["task"]),
            "mounted_read_only_for_live_evaluation": True,
        })
        write_new(run / "campaign/task-workloads.json", {
            "task": assignment["task"],
            "workload_ids": list(workload_ids[assignment["task"]]),
        })
        write_new(run / "campaign/budget.json", {
            "total_wall_seconds": author["wall_time_seconds"],
            "search_wall_seconds": author["wall_time_seconds"] - author["confirmation_seconds"],
            "confirmation_wall_seconds": author["confirmation_seconds"],
            "tokens": "accounting_only",
        })
        (run / "campaign/JOURNAL.md").write_text("# Own C550 Bench search journal\n")
        with (run / ".gitignore").open("a") as stream:
            stream.write("\n# Per-Run campaign evidence stays outside source control.\n")
            for name in GENERATED:
                stream.write(name + "\n")
        subprocess.run(
            [sys.executable, str(run / "campaign/hmz_compiler_tools.py"), "freeze"],
            cwd=run, check=True, stdout=subprocess.DEVNULL,
        )
        subprocess.run([
            "git", "-C", str(run), "add", ".gitignore", "campaign/frozen-plan.json",
            "campaign/frozen-routing.json", "campaign/binding.json",
            "campaign/data-binding.json", "campaign/task-workloads.json",
            "campaign/budget.json", "campaign/hmz_compiler_tools.py",
            "campaign/compiler-api.json",
        ], check=True)
        subprocess.run([
            "git", "-C", str(run), "commit", "-qm",
            "Prepare frozen whole-task C550 Bench allocation",
        ], check=True)
        records.append({**binding, "prepared_commit": git(run, "rev-parse", "HEAD")})
    allocation = {
        "schema_version": 1,
        "kind": "c550_bench_host_allocation",
        "host": host,
        "start_window_seconds": config["start_window_seconds"],
        "runs": records,
        "scope": "prepared without provider or GPU; live MetaX adapter not qualified",
    }
    write_new(output, allocation)
    return allocation
