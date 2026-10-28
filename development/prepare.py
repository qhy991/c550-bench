"""Prepare fresh external-HMZ workspaces; no provider or device work."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .binding import clean_checkout, git, read, render_task, validate

GENERATED = (
    "campaign/evaluation-requests/",
    "campaign/compiler-actions/", "campaign/evaluations/", "campaign/native-engineering/",
    "campaign/management/", "campaign/logs/", "campaign/baseline-preparation/",
    "campaign/runtime/", "campaign/prepared-baseline.json", "campaign/runtime.json",
    "campaign/authority.json", "campaign/native-interface.json",
    "campaign/controller-status.json", "campaign/deadline.json",
    "campaign/intake.json", "campaign/ENDPOINT.json", "campaign/DONE.json",
    "campaign/SEARCH_CLOSE.json", "campaign/search-close-owner.json",
)


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def clone(source: Path, destination: Path, commit: str) -> None:
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(source), str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--quiet", "--detach", commit], check=True)


def source_identity(root: Path) -> dict[str, str]:
    identity = {}
    for key in ("user.name", "user.email"):
        value = git(root, "config", "--get", key)
        if not value:
            raise ValueError(f"configure launcher checkout Git {key} before preparation")
        identity[key] = value
    return identity


ADMISSION = """
import json, sys
from pathlib import Path
from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.tasks.workloads import load_workload
root = Path(sys.argv[1]); compiler = Compiler.load(root, root / "compiler/revision.json")
refused = {}
for task, workload, starter in json.loads(sys.argv[2]):
    try:
        if load_workload(Path(workload)).target != "xcore1000":
            raise ValueError("Workload target is not xcore1000")
        if not compiler.assess(frontend.read_schedule(Path(starter)).document).lowering_eligible:
            raise ValueError("starter is not lowering eligible")
    except Exception as error:
        refused[task] = type(error).__name__ + ": " + str(error)
print(json.dumps(refused))
"""


def require_pinned_compiler_admits(compiler_root: Path, rows) -> None:
    """Refuse before any workspace exists when the pinned Compiler cannot read a task.

    A task newer than the pin (for example aka_colwise_lhs_broadcast_subtract before
    open-cake-ir #789) would otherwise fail inside the live Run's own preparation.
    """
    sources = [[row["task"], row["source"]["workload"], row["source"]["starter"]] for row in rows]
    result = subprocess.run([sys.executable, "-c", ADMISSION, str(compiler_root), json.dumps(sources)],
                            capture_output=True, text=True, cwd=compiler_root,
                            env={**os.environ, "PYTHONPATH": str(compiler_root / "src"),
                                 "PYTHONDONTWRITEBYTECODE": "1"})
    if result.returncode != 0:
        raise ValueError("pinned Compiler admission check failed: " + result.stderr.strip()[-400:])
    refused = json.loads(result.stdout)
    if refused:
        raise ValueError("pinned Compiler refuses these tasks' retained sources: "
                         + json.dumps(refused, sort_keys=True))


def prepare(plan_path: Path, host: str, output: Path, *, repository: Path | None = None,
            tasks: frozenset[str] | None = None) -> dict:
    repository = Path(repository or Path(__file__).resolve().parents[1]).resolve(strict=True)
    plan = read(plan_path)
    rows = validate(plan, repository)
    config = plan["hosts"].get(host)
    selected = [row for row in rows if row["host"] == host]
    if tasks is not None:
        if not tasks or not tasks.issubset({row['task'] for row in selected}):
            raise ValueError('selected tasks differ from the frozen host assignments')
        selected = [row for row in selected if row['task'] in tasks]
    if config is None or not selected:
        raise ValueError("requested host has no frozen assignments")
    adapter_root = clean_checkout(Path(config["adapter_root"]), plan["adapter_commit"])
    compiler_root = clean_checkout(Path(config["compiler_root"]), plan["compiler_commit"])
    identity = source_identity(adapter_root)
    for relative in plan["skills"].values():
        path = compiler_root / relative
        if not path.is_file() or path.resolve() != path:
            raise ValueError(f"required author skill is unavailable: {relative}")
    adapter = compiler_root / "tools/hmz_compiler_tools.py"
    if not adapter.is_file():
        raise ValueError("pinned Compiler has no HMZ tool adapter")
    output = Path(output).resolve()
    if output.exists() or any((parent / ".git").exists() for parent in output.parents):
        raise ValueError("allocation output must be fresh and outside a checkout")
    require_pinned_compiler_admits(compiler_root, selected)
    prepared = []
    for sequence, row in enumerate(selected, 1):
        run = Path(row["root"])
        if run.exists():
            raise FileExistsError(f"fresh development workspace exists: {run}")
        run.parent.mkdir(parents=True, exist_ok=True)
        clone(adapter_root, run, plan["adapter_commit"])
        for key, value in identity.items():
            subprocess.run(["git", "-C", str(run), "config", "--local", key, value], check=True)
        (run / ".deps").mkdir()
        (run / ".deps/cake-ir").symlink_to(compiler_root, target_is_directory=True)
        (run / ".deps/launcher").symlink_to(adapter_root, target_is_directory=True)
        for name in ("candidates", "compiler-actions", "evaluations", "native-engineering",
                     "management", "logs"):
            (run / "campaign" / name).mkdir(parents=True, exist_ok=True)
        source = row["source"]
        for key in ("workload", "starter", "prior_run", "prepared_baseline"):
            if not Path(source[key]).is_file():
                raise ValueError(f"retained {key} is unavailable for {row['task']}")
        shutil.copyfile(source["workload"], run / "campaign/original-workload.json")
        shutil.copyfile(source["starter"], run / "campaign/candidates/starter.py")
        shutil.copyfile(source["prior_run"], run / "campaign/prior-run.json")
        shutil.copyfile(adapter, run / "campaign/compiler_tools.py")
        binding = {
            "schema_version": 1,
            "kind": "c500_hmz_development",
            "sequence": sequence,
            "task": row["task"],
            "host": host,
            "home": config["home"],
            "root": str(run),
            "compiler": plan["compiler_commit"],
            "adapter_commit": plan["adapter_commit"],
            "target": plan["target"],
            "device": row["device"],
            **plan["controls"],
            "skills": plan["skills"],
            "source_provenance": source,
            "routes": {
                "default": "cake",
                "fallback": "separate_native_triton_engineering",
                "route_switch_inside_candidate": False,
            },
        }
        write_new(run / "campaign/development-binding.json", binding)
        write_new(run / "campaign/frozen-plan.json", plan)
        (run / "campaign/TASK.md").write_text(
            render_task(repository, task=row["task"], compiler=plan["compiler_commit"])
        )
        (run / "campaign/AGENTS.md").write_text(
            "# C500 HMZ development author\n\n"
            "Read campaign/TASK.md and both named skills before editing a candidate.\n"
            "Use only this task's HMZ Compiler tools and campaign evaluator.\n"
            "Keep Cake and separate native-route evidence distinct.\n"
        )
        (run / "campaign/JOURNAL.md").write_text("# Own task journal\n")
        (run / "campaign/EVOLUTION.md").write_text("# Mechanisms and promotion disposition\n")
        with (run / ".gitignore").open("a") as stream:
            stream.write("\n.deps/\n")
            for path in GENERATED:
                stream.write(path + "\n")
        subprocess.run([sys.executable, "campaign/compiler_tools.py", "freeze"],
                       cwd=run, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "-C", str(run), "add", ".gitignore", "campaign"], check=True)
        subprocess.run(["git", "-C", str(run), "commit", "-qm",
                        "Prepare frozen C500 HMZ development task"], check=True)
        prepared.append({**binding, "prepared_commit": git(run, "rev-parse", "HEAD")})
    allocation = {
        "schema_version": 1,
        "kind": "c500_hmz_development_allocation",
        "host": host,
        "compiler": plan["compiler_commit"],
        "adapter_commit": plan["adapter_commit"],
        "runs": prepared,
        "scope": "prepared without provider or GPU; live intake remains separately qualified",
    }
    write_new(output, allocation)
    return allocation
