#!/usr/bin/env python3
"""Freeze retained MetaX C500 task identities into one external-HMZ development plan.

Each task's Workload, starter, Run specification and prepared baseline come from its
newest complete retained C500 Run under --retained (for example /var/tmp/cake-c500-bench). A task
without all four is reported and left out of the denominator, never silently dropped.
"""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
HOST = "metax-infini"
PCIS = ("0000:09:00", "0000:0e:00", "0000:1c:00", "0000:20:00")
SOURCE = {"workload": "workload.json", "starter": "starter.py", "prior_run": "run.json",
          "prepared_baseline": "baseline/candidate.json"}


def retained(directory, tasks):
    rows, missing = [], {}
    for task in tasks:
        runs = sorted((p for p in Path(directory).glob(task + "-00[0-9]") if p.is_dir()), reverse=True)
        # A Run that stopped before its specification was written leaves no run.json; the
        # newest Run that did is the task's frozen source.
        complete = [run for run in runs if all((run / name).is_file() for name in SOURCE.values())]
        if not complete:
            missing[task] = "no retained C500 Run has all of " + ", ".join(SOURCE.values())
            continue
        run = complete[0]
        source = {name: run / relative for name, relative in SOURCE.items()}
        if json.loads(source["prior_run"].read_text())["execution"].get("target") != "xcore1000":
            missing[task] = f"{run.name} is not an xcore1000 Run"
            continue
        rows.append({"task": task, **{name: str(path.resolve()) for name, path in source.items()}})
    return rows, missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--retained", type=Path, required=True)
    p.add_argument("--compiler-commit", required=True)
    p.add_argument("--adapter-commit", required=True)
    for name in ("compiler-root", "adapter-root", "home", "owner-root"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if any(not re.fullmatch(r"[0-9a-f]{40}", x) for x in (a.compiler_commit, a.adapter_commit)):
        raise ValueError("commits must be full lowercase Git identities")
    catalog = json.loads((ROOT / "development/tasks.json").read_text())
    rows, missing = retained(a.retained, catalog["tasks"])
    if missing:
        raise ValueError("retained C500 sources are incomplete: " + json.dumps(missing, sort_keys=True))
    if not Path(a.owner_root).is_relative_to(a.home):
        raise ValueError("hmz fences writes only for Runs under the home directory")
    devices = [{"physical_device": i, "runtime_device": i, "expected_pci": pci} for i, pci in enumerate(PCIS)]
    host = {"compiler_root": a.compiler_root, "adapter_root": a.adapter_root, "home": a.home,
            "owner_root": a.owner_root, "devices": devices}
    assignments = [{"task": row["task"], "host": HOST, "root": str(Path(a.owner_root) / row["task"]),
                    "device": devices[index % len(devices)],
                    "source": {key: row[key] for key in ("workload", "starter", "prior_run", "prepared_baseline")}}
                   for index, row in enumerate(rows)]
    document = {"schema_version": 1, "kind": "c500_hmz_development_round", "target": "xcore1000",
        "compiler_commit": a.compiler_commit, "adapter_commit": a.adapter_commit,
        "controls": {"model": "glm-5.3", "reasoning_effort": "high", "wall_time_seconds": 10800,
            "search_seconds": 9720, "confirmation_seconds": 1080, "tokens": "accounting_only",
            "authors_per_device": 1, "reference_access": "known_kernel_reproduction"},
        "skills": {"evolve": "skills/evolve/SKILL.md",
                   "metax-c550-hardware-exploration": "skills/metax-c550-hardware-exploration/SKILL.md"},
        "hosts": {HOST: host}, "assignments": assignments, "unsupported": catalog["unsupported"],
        "stopping_rule": "all assignments classified; no replacement Runs or budget reset"}
    with a.output.open("x") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"tasks": len(assignments), "output": str(a.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
