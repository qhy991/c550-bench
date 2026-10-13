"""Validate one static 54-task C550 HMZ development allocation."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

KIND = "c550_hmz_development_round"
TARGET = "xcore1002"
SKILLS = {
    "evolve": "skills/evolve/SKILL.md",
    "metax-c550-hardware-exploration": "skills/metax-c550-hardware-exploration/SKILL.md",
}


def read(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected an object: {path}")
    return value


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE, timeout=30
    ).strip()


def clean_checkout(root: Path, commit: str) -> Path:
    root = Path(root).resolve(strict=True)
    if git(root, "rev-parse", "--show-toplevel") != str(root):
        raise ValueError("pass the checkout root")
    if git(root, "rev-parse", "HEAD") != commit:
        raise ValueError("checkout differs from the frozen commit")
    if git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("checkout is not clean")
    return root


def task_ids(root: Path) -> tuple[str, ...]:
    document = read(root / "development/tasks.json")
    tasks = document.get("tasks")
    unsupported = document.get("unsupported")
    if (document.get("schema_version") != 1
            or document.get("kind") != "c550_registered_development_tasks"
            or not isinstance(tasks, list) or len(tasks) != 54
            or len(set(tasks)) != 54
            or any(not isinstance(task, str) or not re.fullmatch(r"[a-z0-9_]+", task)
                   for task in tasks)
            or unsupported != [{"task": "tinygemm", "status": "unsupported_target",
                                "target": TARGET}]):
        raise ValueError("development task denominator differs")
    return tuple(tasks)


def validate(plan: dict, repository: Path) -> tuple[dict, ...]:
    expected_tasks = set(task_ids(repository))
    if (plan.get("schema_version") != 1 or plan.get("kind") != KIND
            or plan.get("target") != TARGET):
        raise ValueError("unsupported C550 development plan")
    for field in ("compiler_commit", "adapter_commit"):
        if not isinstance(plan.get(field), str) or not re.fullmatch(r"[0-9a-f]{40}", plan[field]):
            raise ValueError(f"plan requires an immutable {field}")
    controls = plan.get("controls")
    if not isinstance(controls, dict) or controls != {
        "model": "glm-5.3", "reasoning_effort": "high",
        "wall_time_seconds": 10800, "search_seconds": 9720,
        "confirmation_seconds": 1080, "tokens": "accounting_only",
        "authors_per_device": 2, "reference_access": "known_kernel_reproduction",
    }:
        raise ValueError("C550 development controls differ")
    if plan.get("skills") != SKILLS:
        raise ValueError("plan must grant the exact evolve and MetaX skills")
    hosts = plan.get("hosts")
    if not isinstance(hosts, dict) or set(hosts) != {"c550-1", "c550-2"}:
        raise ValueError("plan requires the two declared C550 hosts")
    for host, config in hosts.items():
        if not isinstance(config, dict):
            raise ValueError("host configuration must be an object")
        for name in ("compiler_root", "adapter_root", "home", "owner_root"):
            if not isinstance(config.get(name), str) or not Path(config[name]).is_absolute():
                raise ValueError(f"{host} requires absolute {name}")
        devices = config.get("devices")
        if (not isinstance(devices, list) or len(devices) != 8
                or {d.get("physical_device") for d in devices} != set(range(8))
                or any(d.get("runtime_device") != 0 or not isinstance(d.get("expected_pci"), str)
                       or not d["expected_pci"] for d in devices)):
            raise ValueError(f"{host} requires eight exact physical/runtime/PCI routes")
    rows = plan.get("assignments")
    if (not isinstance(rows, list) or len(rows) != 54
            or {row.get("task") for row in rows} != expected_tasks):
        raise ValueError("plan must assign the intact 54-task denominator")
    roots = set()
    for row in rows:
        host = row.get("host")
        if host not in hosts or row.get("task") not in expected_tasks:
            raise ValueError("assignment names a foreign host or task")
        root = Path(row.get("root", ""))
        if not root.is_absolute() or root != root.resolve() or str(root) in roots:
            raise ValueError("each assignment requires a unique canonical root")
        roots.add(str(root))
        source = row.get("source")
        if not isinstance(source, dict) or set(source) != {
                "workload", "starter", "prior_run", "prepared_baseline"}:
            raise ValueError("assignment source fields differ")
        if any(not isinstance(value, str) or not Path(value).is_absolute()
               for value in source.values()):
            raise ValueError("assignment source paths must be absolute")
        device = row.get("device")
        declared = {d["physical_device"]: d for d in hosts[host]["devices"]}
        if not isinstance(device, dict) or device != declared.get(device.get("physical_device")):
            raise ValueError("assignment device differs from the host route")
    return tuple(rows)


def render_task(repository: Path, *, task: str, compiler: str) -> str:
    template = (repository / "development/TASK.md").read_text()
    return template.replace("{task}", task).replace("{compiler}", compiler)
