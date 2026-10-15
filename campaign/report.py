"""Project every frozen allocation and workload from create-only campaign evidence."""

from __future__ import annotations

from pathlib import Path
import re

from .binding import read_object, reconcile_allocation


def _read(path: Path) -> tuple[dict | None, str | None]:
    try:
        value = read_object(path)
        return value, None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError, TypeError) as error:
        return None, f"unreadable:{type(error).__name__}"


def _evidence(value: object, missing: str = "missing") -> dict:
    if isinstance(value, dict) and isinstance(value.get("status"), str):
        return value
    return {"status": missing}


def _workloads(run: dict, terminal: dict | None) -> list[dict]:
    observed = terminal.get("workloads", {}) if isinstance(terminal, dict) else {}
    if not isinstance(observed, dict):
        observed = {}
    rows = []
    for identity in run["workload_ids"]:
        value = observed.get(identity)
        if not isinstance(value, dict):
            value = {}
        rows.append({
            "uuid": identity,
            "correctness": _evidence(value.get("correctness"), "unexamined"),
            "measurement": _evidence(value.get("measurement"), "unexamined"),
            "profile": _evidence(value.get("profile"), "unexamined"),
            "confirmation": _evidence(value.get("confirmation"), "unexamined"),
        })
    return rows


def report(allocation_path: Path, state_directory: Path | None = None) -> dict:
    allocation = read_object(allocation_path)
    if allocation.get("schema_version") != 1 or allocation.get("kind") != "c550_bench_host_allocation":
        raise ValueError("unsupported C550 host allocation")
    reconcile_allocation(allocation)
    state = Path(state_directory) if state_directory is not None else None
    rows = []
    for run in allocation.get("runs", []):
        identity = run.get("assignment_id")
        if (not isinstance(identity, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", identity)
                or len(run.get("workload_ids", [])) != 16):
            raise ValueError("prepared allocation binding is malformed")
        claim, claim_error = _read(state / "claims" / f"{identity}.json") if state else (None, "missing")
        terminal, terminal_error = _read(state / "terminals" / f"{identity}.json") if state else (None, "missing")
        release, release_error = _read(state / "releases" / f"{identity}.json") if state else (None, "missing")
        review, review_error = _read(state / "reviews" / f"{identity}.json") if state else (None, "missing")
        if terminal is not None:
            execution = _evidence(terminal.get("execution"), "unknown")
        elif claim is not None:
            execution = {"status": "unknown", "reason": "claimed_without_terminal"}
        else:
            execution = {"status": "prepared"}
        release_projection = (_evidence(release, "unverified") if release is not None
                              else {"status": "unverified" if claim else "not_started"})
        review_projection = (_evidence(review, "missing") if review is not None
                             else {"status": "missing"})
        if review is not None and release_projection.get("status") != "verified":
            review_projection = {"status": "invalid", "reason": "review_without_verified_release"}
        rows.append({
            "assignment_id": identity,
            "task": run["task"],
            "condition": run["condition"],
            "replicate": run["replicate"],
            "device_id": run["device_id"],
            "execution": execution,
            "correctness": _evidence((terminal or {}).get("correctness")),
            "performance": _evidence((terminal or {}).get("performance"), "coverage_missing"),
            "confirmation": _evidence((terminal or {}).get("confirmation")),
            "profile": _evidence((terminal or {}).get("profile")),
            "release": release_projection,
            "review": review_projection,
            "workloads": _workloads(run, terminal),
            "evidence_read_errors": {
                key: error for key, error in {
                    "claim": claim_error, "terminal": terminal_error,
                    "release": release_error, "review": review_error,
                }.items() if error not in (None, "missing")
            },
        })
    expected = len(allocation.get("runs", []))
    return {
        "schema_version": 1,
        "kind": "c550_bench_campaign_report",
        "host": allocation.get("host"),
        "assigned": expected,
        "rows": rows,
        "complete_denominator": len(rows) == expected,
        "performance_claim": False,
        "scope": "evidence projection; no device replay or automatic Compiler adoption",
    }
