"""Finite at-most-once author sequencing behind one lifecycle owner."""

from __future__ import annotations

from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Protocol

from .binding import git, read_object
from .report import report


def publish_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".campaign-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"create-only campaign record already exists: {path}") from None
    finally:
        temporary.unlink()
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class AuthorHandle(Protocol):
    def poll(self) -> int | None: ...


class AuthorLauncher(Protocol):
    def start(self, run: dict, claim: dict) -> AuthorHandle: ...
    def terminal(self, run: dict, handle: AuthorHandle, exit_code: int) -> dict: ...


class ReleaseVerifier(Protocol):
    def observe(self, run: dict, claim: dict, terminal: dict) -> dict: ...


class CampaignOwner:
    """Own claims and terminal publication; adapters own provider and device protocols."""

    def __init__(self, allocation: Path, state_directory: Path, *, clock=time.time,
                 sleeper=time.sleep):
        self.allocation_path = Path(allocation).resolve(strict=True)
        self.state = Path(state_directory)
        self.clock = clock
        self.sleeper = sleeper
        self.document = read_object(self.allocation_path)
        if (self.document.get("schema_version") != 1
                or self.document.get("kind") != "c550_bench_host_allocation"):
            raise ValueError("unsupported C550 host allocation")
        runs = self.document.get("runs")
        identities = [row.get("assignment_id") for row in runs or []]
        if (not runs or len(set(identities)) != len(identities)
                or any(not isinstance(value, str)
                       or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value)
                       for value in identities)):
            raise ValueError("host allocation has invalid assignment identities")
        capacities = {}
        for run in runs:
            capacity = run.get("author_capacity")
            if type(capacity) is not int or not 1 <= capacity <= 3:
                raise ValueError("assignment has invalid author capacity")
            old = capacities.setdefault(run["device_id"], capacity)
            if old != capacity:
                raise ValueError("one physical device has conflicting capacities")
            root = Path(run.get("root", ""))
            if not root.is_absolute() or not root.is_dir():
                raise ValueError("assignment prepared workspace is missing")
            if git(root, "rev-parse", "HEAD") != run.get("prepared_commit"):
                raise ValueError("prepared source commit changed")
            binding = read_object(root / "campaign/binding.json")
            if binding != {key: value for key, value in run.items() if key != "prepared_commit"}:
                raise ValueError("prepared binding differs from host allocation")
        self.capacities = capacities

    def _terminal(self, run: dict, terminal: dict) -> None:
        if terminal.get("assignment_id") != run["assignment_id"]:
            raise ValueError("launcher terminal belongs to another assignment")
        publish_new(self.state / "terminals" / f"{run['assignment_id']}.json", terminal)

    def _release(self, run: dict, release: dict) -> None:
        if (release.get("assignment_id") != run["assignment_id"]
                or release.get("device_id") != run["device_id"]
                or release.get("status") not in {"verified", "unverified"}):
            raise ValueError("release verifier returned an invalid result")
        publish_new(self.state / "releases" / f"{run['assignment_id']}.json", release)

    def run(self, launcher: AuthorLauncher, release_verifier: ReleaseVerifier) -> dict:
        self.state.mkdir(parents=True, exist_ok=True)
        with (self.state / "owner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            started = self.clock()
            stop_new = started + self.document["start_window_seconds"]
            publish_new(self.state / "LAUNCH.json", {
                "owner_pid": os.getpid(),
                "started_at_epoch": started,
                "stop_new_at_epoch": stop_new,
                "maximum_runs": len(self.document["runs"]),
                "no_author_restarts": True,
            })
            pending = list(self.document["runs"])
            active: dict[str, tuple[dict, dict, AuthorHandle]] = {}
            blocked_devices = set()
            while True:
                for identity, (run, claim, handle) in list(active.items()):
                    code = handle.poll()
                    if code is None:
                        continue
                    try:
                        terminal = launcher.terminal(run, handle, code)
                    except Exception as error:
                        terminal = {
                            "assignment_id": identity,
                            "execution": {"status": "unknown", "reason": f"terminal:{type(error).__name__}"},
                            "correctness": {"status": "missing"},
                            "performance": {"status": "coverage_missing"},
                            "confirmation": {"status": "missing"},
                            "workloads": {},
                        }
                    self._terminal(run, terminal)
                    try:
                        release = release_verifier.observe(run, claim, terminal)
                    except Exception as error:
                        release = {
                            "assignment_id": identity,
                            "device_id": run["device_id"],
                            "status": "unverified",
                            "reason": f"release:{type(error).__name__}",
                        }
                    self._release(run, release)
                    if release["status"] != "verified":
                        blocked_devices.add(run["device_id"])
                    del active[identity]
                counts = Counter(run["device_id"] for run, _, _ in active.values())
                if self.clock() < stop_new:
                    while True:
                        index = next((index for index, run in enumerate(pending)
                                      if run["device_id"] not in blocked_devices
                                      and counts[run["device_id"]] < self.capacities[run["device_id"]]), None)
                        if index is None:
                            break
                        run = pending[index]
                        device = run["device_id"]
                        now = self.clock()
                        total = run["author"]["wall_time_seconds"]
                        confirmation = run["author"]["confirmation_seconds"]
                        claim = {
                            "assignment_id": run["assignment_id"],
                            "device_id": device,
                            "claimed_at_epoch": now,
                            "search_stop_at_epoch": now + total - confirmation,
                            "stop_at_epoch": now + total,
                            "author_opportunity_consumed": True,
                        }
                        publish_new(self.state / "claims" / f"{run['assignment_id']}.json", claim)
                        pending.pop(index)
                        try:
                            handle = launcher.start(run, claim)
                        except Exception as error:
                            terminal = {
                                "assignment_id": run["assignment_id"],
                                "execution": {"status": "unknown", "reason": f"launch:{type(error).__name__}"},
                                "correctness": {"status": "missing"},
                                "performance": {"status": "coverage_missing"},
                                "confirmation": {"status": "missing"},
                                "workloads": {},
                            }
                            self._terminal(run, terminal)
                            try:
                                release = release_verifier.observe(run, claim, terminal)
                            except Exception as release_error:
                                release = {
                                    "assignment_id": run["assignment_id"],
                                    "device_id": device,
                                    "status": "unverified",
                                    "reason": f"release:{type(release_error).__name__}",
                                }
                            self._release(run, release)
                            if release["status"] != "verified":
                                blocked_devices.add(device)
                        else:
                            active[run["assignment_id"]] = (run, claim, handle)
                            counts[device] += 1
                if not pending and not active:
                    break
                if not active and (self.clock() >= stop_new
                                   or all(run["device_id"] in blocked_devices for run in pending)):
                    reason = ("unstarted_window_expired" if self.clock() >= stop_new
                              else "unstarted_release_blocked")
                    for run in pending:
                        self._terminal(run, {
                            "assignment_id": run["assignment_id"],
                            "execution": {"status": reason},
                            "correctness": {"status": "missing"},
                            "performance": {"status": "coverage_missing"},
                            "confirmation": {"status": "missing"},
                            "workloads": {},
                        })
                    pending.clear()
                    break
                self.sleeper(0.01)
            return report(self.allocation_path, self.state)


def record_review(allocation: Path, state_directory: Path, assignment_id: str,
                  review: dict) -> None:
    state = Path(state_directory)
    document = read_object(allocation)
    if assignment_id not in {row.get("assignment_id") for row in document.get("runs", [])}:
        raise ValueError("review names a foreign assignment")
    terminal = read_object(state / "terminals" / f"{assignment_id}.json")
    release = read_object(state / "releases" / f"{assignment_id}.json")
    if release.get("status") != "verified":
        raise ValueError("independent review requires verified canonical release")
    if terminal.get("assignment_id") != assignment_id:
        raise ValueError("terminal belongs to another assignment")
    if review.get("assignment_id") != assignment_id or review.get("status") not in {
            "accepted", "rejected", "unresolved"}:
        raise ValueError("invalid independent review")
    publish_new(state / "reviews" / f"{assignment_id}.json", review)


def claim_confirmation(state_directory: Path, assignment_id: str, candidate: dict,
                       *, now: float) -> dict:
    """Consume one confirmation opportunity before any confirmation side effect."""
    state = Path(state_directory)
    run_claim = read_object(state / "claims" / f"{assignment_id}.json")
    if (candidate.get("status") != "eligible" or not isinstance(candidate.get("id"), str)
            or not candidate["id"] or type(candidate.get("completed_at_epoch")) not in (int, float)
            or candidate["completed_at_epoch"] > run_claim["search_stop_at_epoch"]
            or now >= run_claim["stop_at_epoch"]):
        raise ValueError("confirmation requires one fixed nominee inside the original budget")
    claim = {"assignment_id": assignment_id, "nominee": candidate["id"],
             "claimed_at_epoch": now, "stop_at_epoch": run_claim["stop_at_epoch"],
             "confirmation_opportunity_consumed": True}
    publish_new(state / "confirmations" / f"{assignment_id}.json", claim)
    return claim
