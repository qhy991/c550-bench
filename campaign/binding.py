"""Admit one frozen evolve plan and its explicit C550 host routing."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess


PLAN_KIND = "rolling_bench_fresh_search"
ROUTING_KIND = "c550_bench_routing"
TARGET = "metax_c550"
CONDITIONS = ("control", "successor")


def read_object(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments], text=True, stderr=subprocess.PIPE,
        timeout=30,
    ).strip()


def clean_checkout(root: Path, commit: str) -> Path:
    root = Path(root).resolve(strict=True)
    if git(root, "rev-parse", "--show-toplevel") != str(root):
        raise ValueError("pass a checkout root")
    if git(root, "rev-parse", "HEAD") != commit:
        raise ValueError(f"checkout differs from frozen commit: {root}")
    if git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError(f"checkout is not clean: {root}")
    return root


def tracked_file(root: Path, relative: str, label: str) -> Path:
    if (not isinstance(relative, str) or not relative
            or Path(relative).is_absolute() or ".." in Path(relative).parts):
        raise ValueError(f"{label} must be a repository-relative file")
    git(root, "cat-file", "-e", f"HEAD:{relative}")
    path = root / relative
    if not path.is_file() or path.resolve() != path:
        raise ValueError(f"{label} is not a regular tracked file: {relative}")
    return path


def task_workload_ids(data_root: Path, task_id: str) -> tuple[str, ...]:
    task_root = data_root / "benchmark" / task_id
    for name in ("definition.json", "reference.py", "workload.jsonl"):
        if not (task_root / name).is_file():
            raise ValueError(f"prepared task data is incomplete: {task_id}")
    rows = [json.loads(line) for line in (task_root / "workload.jsonl").read_text().splitlines()
            if line.strip()]
    identities = tuple(row.get("uuid") for row in rows)
    if (len(identities) != 16 or len(set(identities)) != 16
            or any(not isinstance(value, str) or not value for value in identities)):
        raise ValueError(f"task must retain sixteen unique original workload UUIDs: {task_id}")
    return identities


def validate_plan(plan: dict, bench_root: Path, data_root: Path) -> dict[str, tuple[str, ...]]:
    if plan.get("schema_version") != 1 or plan.get("kind") != PLAN_KIND:
        raise ValueError("unsupported frozen plan; use open-cake-ir evolve freeze-bench")
    if plan.get("target") != TARGET:
        raise ValueError("frozen plan target is not MetaX C550")
    bindings = plan.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != {"bench", *CONDITIONS}:
        raise ValueError("frozen plan bindings differ")
    bench_commit = bindings["bench"].get("commit")
    if not isinstance(bench_commit, str):
        raise ValueError("frozen Bench commit is missing")
    clean_checkout(bench_root, bench_commit)
    tracked_file(bench_root, "suite.json", "suite")
    suite = read_object(bench_root / "suite.json")
    if suite.get("version") != 2 or suite.get("target_arch") != TARGET:
        raise ValueError("campaign requires C550 Bench suite v2")
    suite_tasks = [row.get("id") for row in suite.get("tasks", [])]
    tasks = plan.get("task_ids")
    if (not isinstance(tasks, list) or not tasks or len(set(tasks)) != len(tasks)
            or any(task not in suite_tasks for task in tasks)):
        raise ValueError("frozen task list differs from the pinned suite")
    receipt = read_object(data_root / "materialization-suite-v2.json")
    if receipt.get("sources") != read_object(bench_root / "sources.lock.json"):
        raise ValueError("prepared task data differs from the pinned sources")
    receipt_rows = receipt.get("tasks")
    if (not isinstance(receipt_rows, list)
            or {row.get("task") for row in receipt_rows} != set(suite_tasks)
            or len(receipt_rows) != len(suite_tasks)
            or any(row.get("workloads") != 16 for row in receipt_rows)):
        raise ValueError("prepared task-data receipt does not cover suite v2")
    author = plan.get("author")
    if not isinstance(author, dict) or set(author) != {
            "model", "scaffold", "knowledge", "reference_access",
            "wall_time_seconds", "confirmation_seconds"}:
        raise ValueError("frozen author controls differ")
    if any(not isinstance(author[key], str) or not author[key].strip()
           for key in ("model", "scaffold", "knowledge", "reference_access")):
        raise ValueError("frozen author selections must be explicit")
    tracked_file(bench_root, author["scaffold"], "author scaffold")
    wall, confirmation = author["wall_time_seconds"], author["confirmation_seconds"]
    if type(wall) is not int or type(confirmation) is not int or not 0 < confirmation < wall:
        raise ValueError("one total wall budget must include a positive confirmation reserve")
    baselines = plan.get("baseline_files")
    if not isinstance(baselines, dict) or set(baselines) != set(tasks):
        raise ValueError("every task requires one fixed baseline")
    tracked_file(bench_root, plan.get("measurement_protocol"), "measurement protocol")
    for task, relative in baselines.items():
        tracked_file(bench_root, relative, f"baseline for {task}")
    allocations = plan.get("allocations")
    replicates = plan.get("replicates")
    if type(replicates) is not int or replicates < 1:
        raise ValueError("frozen replicates must be a positive integer")
    expected = len(tasks) * 2 * replicates
    if not isinstance(allocations, list) or len(allocations) != expected:
        raise ValueError("frozen allocation does not cover every task, condition and replicate")
    seen = set()
    for row in allocations:
        key = (row.get("task"), row.get("condition"), row.get("replicate"))
        if (row.get("task") not in tasks or row.get("condition") not in CONDITIONS
                or type(row.get("replicate")) is not int or not 1 <= row["replicate"] <= replicates
                or not isinstance(row.get("workspace"), str) or not Path(row["workspace"]).is_absolute()
                or key in seen):
            raise ValueError("invalid or duplicate frozen allocation row")
        seen.add(key)
    ceiling = plan.get("total_author_wall_seconds")
    if type(ceiling) is not int or ceiling < expected * wall:
        raise ValueError("total author budget cannot fund every frozen allocation")
    suite_by_id = {row["id"]: row for row in suite["tasks"]}
    identities = {task: task_workload_ids(data_root, task) for task in tasks}
    if any(suite_by_id[task]["smoke_workload_uuid"] not in identities[task] for task in tasks):
        raise ValueError("prepared task data omits a frozen smoke workload")
    return identities


def validate_routing(plan: dict, routing: dict) -> dict[str, dict]:
    if routing.get("schema_version") != 1 or routing.get("kind") != ROUTING_KIND:
        raise ValueError("unsupported C550 routing document")
    hosts = routing.get("hosts")
    routes = routing.get("assignments")
    if not isinstance(hosts, dict) or not hosts or not isinstance(routes, list):
        raise ValueError("routing requires hosts and assignments")
    expected = {row["workspace"]: row for row in plan["allocations"]}
    if len(routes) != len(expected):
        raise ValueError("routing must assign every frozen allocation exactly once")
    by_workspace = {}
    physical_roots = set()
    for host, config in hosts.items():
        if not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", host):
            raise ValueError("invalid host identity")
        if not isinstance(config, dict):
            raise ValueError("host configuration must be an object")
        devices = config.get("devices")
        if not isinstance(devices, list) or not devices:
            raise ValueError("host requires explicit C550 devices")
        ids, ordinals = set(), set()
        for device in devices:
            identity, ordinal, capacity = (device.get("id"), device.get("physical_device"),
                                           device.get("author_capacity"))
            if (not isinstance(identity, str) or not identity or identity in ids
                    or type(ordinal) is not int or ordinal < 0 or ordinal in ordinals
                    or type(capacity) is not int or not 1 <= capacity <= 3):
                raise ValueError("invalid or duplicate physical C550 device route")
            ids.add(identity)
            ordinals.add(ordinal)
        if type(config.get("start_window_seconds")) is not int or config["start_window_seconds"] <= 0:
            raise ValueError("host requires a positive start window")
    for route in routes:
        workspace, host, root, device = (route.get("workspace"), route.get("host"),
                                         route.get("root"), route.get("device_id"))
        if workspace not in expected or workspace in by_workspace or host not in hosts:
            raise ValueError("routing contains a foreign or duplicate allocation")
        root = Path(root) if isinstance(root, str) else None
        if root is None or not root.is_absolute() or root != root.resolve() or ".." in root.parts:
            raise ValueError("physical workspace must be a canonical absolute path")
        if (host, str(root)) in physical_roots:
            raise ValueError("physical workspace is assigned twice")
        if device not in {row["id"] for row in hosts[host]["devices"]}:
            raise ValueError("assignment names an undeclared physical device")
        physical_roots.add((host, str(root)))
        by_workspace[workspace] = {**route, "root": str(root), **expected[workspace]}
    if set(by_workspace) != set(expected):
        raise ValueError("routing omits a frozen allocation")
    pairs = {}
    for row in by_workspace.values():
        key = (row["task"], row["replicate"])
        value = (row["host"], row["device_id"])
        if key in pairs and pairs[key] != value:
            raise ValueError("control and successor for one task must share host and physical device")
        pairs[key] = value
    return by_workspace


def host_config(plan: dict, routing: dict, host: str) -> tuple[dict, list[dict]]:
    routes = validate_routing(plan, routing)
    if host not in routing["hosts"]:
        raise ValueError("requested host is absent from routing")
    selected = [routes[row["workspace"]] for row in plan["allocations"]
                if routes[row["workspace"]]["host"] == host]
    if not selected:
        raise ValueError("requested host has no frozen allocations")
    return routing["hosts"][host], selected


def reconcile_allocation(document: dict) -> dict:
    """Resolve the complete assignment set from its committed preparation inputs."""
    anchor = document.get('plan_source')
    if not isinstance(anchor, dict) or set(anchor) != {'root', 'prepared_commit'}:
        raise ValueError('allocation requires its committed frozen plan source')
    root = clean_checkout(Path(anchor['root']), anchor['prepared_commit'])
    plan = json.loads(git(root, 'show', anchor['prepared_commit'] + ':campaign/frozen-plan.json'))
    routing = json.loads(git(root, 'show', anchor['prepared_commit'] + ':campaign/frozen-routing.json'))
    expected_keys = {(task, condition, replicate) for task in plan['task_ids']
                     for condition in CONDITIONS for replicate in range(1, plan['replicates'] + 1)}
    keys = [(row['task'], row['condition'], row['replicate']) for row in plan['allocations']]
    if len(keys) != len(expected_keys) or set(keys) != expected_keys:
        raise ValueError('frozen plan omits or duplicates task, condition or replicate')
    config, selected = host_config(plan, routing, document['host'])
    expected = {Path(row['workspace']).name: row for row in selected}
    actual = {row['assignment_id']: row for row in document.get('runs', [])}
    if (len(actual) != len(document.get('runs', [])) or set(actual) != set(expected)):
        raise ValueError('allocation differs from frozen assignments: missing=' +
                         str(sorted(set(expected) - set(actual))) +
                         ', foreign=' + str(sorted(set(actual) - set(expected))))
    if document['start_window_seconds'] != config['start_window_seconds']:
        raise ValueError('allocation start window differs from frozen routing')
    devices = {row['id']: row for row in config['devices']}
    for identity, assignment in expected.items():
        route = devices[assignment['device_id']]
        fields = {key: assignment[key] for key in ('task', 'condition', 'replicate', 'root', 'host', 'device_id')}
        fields.update(frozen_workspace=assignment['workspace'],
                      physical_device=route['physical_device'], author_capacity=route['author_capacity'],
                      bench_commit=plan['bindings']['bench']['commit'],
                      compiler_commit=plan['bindings'][assignment['condition']]['commit'],
                      author=plan['author'], measurement_protocol=plan['measurement_protocol'],
                      baseline=plan['baseline_files'][assignment['task']])
        if any(actual[identity].get(key) != value for key, value in fields.items()):
            raise ValueError('allocation assignment differs from frozen plan/routing: ' + identity)
    return plan
