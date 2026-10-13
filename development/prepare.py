"""Prepare fresh external-HMZ workspaces; no provider or device work."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

from .binding import clean_checkout, git, read, render_task, validate

GENERATED = (
    "campaign/compiler-actions/", "campaign/evaluations/", "campaign/native-engineering/",
    "campaign/management/", "campaign/logs/", "campaign/baseline-preparation/",
    "campaign/runtime/", "campaign/prepared-baseline.json", "campaign/runtime.json",
    "campaign/authority.json", "campaign/controller-status.json", "campaign/deadline.json",
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


def prepare(plan_path: Path, host: str, output: Path, *, repository: Path | None = None) -> dict:
    repository = Path(repository or Path(__file__).resolve().parents[1]).resolve(strict=True)
    plan = read(plan_path)
    rows = validate(plan, repository)
    config = plan["hosts"].get(host)
    selected = [row for row in rows if row["host"] == host]
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
        for name in ("candidates", "compiler-actions", "evaluations", "native-engineering",
                     "management", "logs"):
            (run / "campaign" / name).mkdir(parents=True, exist_ok=True)
        source = row["source"]
        for key in ("workload", "starter", "prior_run", "prepared_baseline"):
            if not Path(source[key]).is_file():
                raise ValueError(f"retained {key} is unavailable for {row['task']}")
        shutil.copyfile(source["workload"], run / "campaign/original-workload.json")
        shutil.copyfile(source["starter"], run / "campaign/candidates/starter.py")
        shutil.copyfile(adapter, run / "campaign/compiler_tools.py")
        binding = {
            "schema_version": 1,
            "kind": "c550_hmz_development",
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
            "# C550 HMZ development author\n\n"
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
                        "Prepare frozen C550 HMZ development task"], check=True)
        prepared.append({**binding, "prepared_commit": git(run, "rev-parse", "HEAD")})
    allocation = {
        "schema_version": 1,
        "kind": "c550_hmz_development_allocation",
        "host": host,
        "compiler": plan["compiler_commit"],
        "adapter_commit": plan["adapter_commit"],
        "runs": prepared,
        "scope": "prepared without provider or GPU; live intake remains separately qualified",
    }
    write_new(output, allocation)
    return allocation
