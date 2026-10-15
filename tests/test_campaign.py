"""Frozen whole-task campaign preparation, dispatch and denominator behavior."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from campaign.owner import CampaignOwner, claim_confirmation, record_review
from campaign.prepare import prepare
from campaign.report import report


ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def repository(root, files):
    root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Campaign Test"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "campaign@example.invalid"], check=True)
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
    return git(root, "rev-parse", "HEAD")


class PreparationTests(unittest.TestCase):
    def make_inputs(self, directory):
        base = Path(directory).resolve()
        suite = json.loads((ROOT / "suite.json").read_text())
        baselines = {}
        upstream_root = base / "upstream"
        upstream_commit = repository(upstream_root, {"src/sol_execbench/core/__init__.py": ""})
        sources = json.loads((ROOT / "sources.lock.json").read_text())
        sources["sol_execbench"]["revision"] = upstream_commit
        bench_files = {
            "suite.json": json.dumps(suite),
            "sources.lock.json": json.dumps(sources),
            "campaign/protocol.json": json.dumps({"timer": "qualification_required"}),
            "campaign/TASK.md": "# Fixture whole-task author scaffold\n",
            ".gitignore": ".deps/\n.data/\n",
        }
        data = base / "data"
        for task_index, task in enumerate(suite["tasks"]):
            name = f"baselines/task-{task_index:02d}.py"
            baselines[task["id"]] = name
            bench_files[name] = "def run(*inputs):\n    return inputs[0]\n"
            task_root = data / "benchmark" / task["id"]
            task_root.mkdir(parents=True)
            (task_root / "definition.json").write_text("{}")
            (task_root / "reference.py").write_text("def run(*inputs): return inputs[0]\n")
            rows = [{"uuid": task["smoke_workload_uuid"]}]
            rows += [{"uuid": f"{task_index:02d}-workload-{index:02d}"} for index in range(1, 16)]
            (task_root / "workload.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        write(data / "materialization-suite-v2.json", {
            "sources": sources,
            "tasks": [{"task": task["id"], "workloads": 16} for task in suite["tasks"]],
            "raw_dataset_in_git": False,
        })
        bench = base / "bench"
        bench_commit = repository(bench, bench_files)
        (bench / ".deps").mkdir()
        subprocess.run(["git", "clone", "-q", str(upstream_root), str(bench / ".deps/sol-execbench")], check=True)
        compilers = {}
        compiler_commits = {}
        for condition in ("control", "successor"):
            source = base / condition
            compiler_commits[condition] = repository(
                source, {"compiler/revision.json": json.dumps({"condition": condition})})
            compilers[condition] = str(source)
        adapter_root = base / "adapter"
        adapter_source = (
            "import json, pathlib, sys\n"
            "root=pathlib.Path.cwd()\n"
            "assert sys.argv[1:]==['freeze']\n"
            "binding=json.loads((root/'campaign/binding.json').read_text())\n"
            "(root/'campaign/compiler-api.json').write_text(json.dumps({'compiler_commit':binding['compiler_commit'],'transformations':[]}))\n"
        )
        adapter_commit = repository(adapter_root, {"hmz_compiler_tools.py": adapter_source})
        adapter = adapter_root / "hmz_compiler_tools.py"
        allocations = []
        routes = []
        run_root = base / "runs"
        for task_index, task in enumerate(suite["tasks"], 1):
            device = f"c550-a/device-{(task_index - 1) % 2}"
            for condition in ("control", "successor"):
                identity = f"t{task_index:03d}-r1-{condition}"
                workspace = str(base / "intent" / "runs" / identity)
                allocations.append({"task": task["id"], "condition": condition,
                                    "replicate": 1, "workspace": workspace})
                routes.append({"workspace": workspace, "host": "c550-a",
                               "root": str((run_root / identity).resolve()), "device_id": device})
        author = {"model": "glm-5.3:high", "scaffold": "campaign/TASK.md",
                  "knowledge": "none", "reference_access": "known_kernel_reproduction",
                  "wall_time_seconds": 10800, "confirmation_seconds": 1800}
        plan = {"schema_version": 1, "kind": "rolling_bench_fresh_search",
                "hypothesis": "fixture", "task_ids": [task["id"] for task in suite["tasks"]],
                "replicates": 1, "allocation_seed": 7, "author": author,
                "measurement_protocol": "campaign/protocol.json", "baseline_files": baselines,
                "total_author_wall_seconds": len(allocations) * 10800,
                "target": "metax_c550",
                "bindings": {"bench": {"root": str(bench), "commit": bench_commit},
                             **{condition: {"root": compilers[condition],
                                            "commit": compiler_commits[condition]}
                                for condition in ("control", "successor")}},
                "allocations": allocations}
        routing = {"schema_version": 1, "kind": "c550_bench_routing",
                   "hosts": {"c550-a": {"bench_root": str(bench), "data_root": str(data),
                       "compiler_roots": compilers,
                       "hmz_adapter": {"path": str(adapter), "source_commit": adapter_commit},
                       "devices": [
                           {"id": "c550-a/device-0", "physical_device": 0, "author_capacity": 1},
                           {"id": "c550-a/device-1", "physical_device": 1, "author_capacity": 1}],
                       "start_window_seconds": 3600}}, "assignments": routes}
        write(base / "plan.json", plan)
        write(base / "routing.json", routing)
        return base / "plan.json", base / "routing.json", base / "allocation.json", run_root

    def test_prepare_all_twelve_whole_tasks_without_provider_or_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, routing, output, _ = self.make_inputs(directory)
            allocation = prepare(plan, routing, "c550-a", output)
            self.assertEqual(len(allocation["runs"]), 24)
            self.assertTrue(all(len(run["workload_ids"]) == 16 for run in allocation["runs"]))
            self.assertEqual({run["task"] for run in allocation["runs"]},
                             {task["id"] for task in json.loads((ROOT / "suite.json").read_text())["tasks"]})
            projected = report(output)
            self.assertEqual(projected["assigned"], 24)
            self.assertTrue(projected["complete_denominator"])
            self.assertTrue(all(len(row["workloads"]) == 16 for row in projected["rows"]))
            self.assertTrue(all(cell["correctness"]["status"] == "unexamined"
                                for row in projected["rows"] for cell in row["workloads"]))

    def test_prepare_refuses_existing_destination_without_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, routing, output, run_root = self.make_inputs(directory)
            prepare(plan, routing, "c550-a", output)
            before = (run_root / "t001-r1-control/campaign/binding.json").read_bytes()
            with self.assertRaises((FileExistsError, ValueError)):
                prepare(plan, routing, "c550-a", Path(directory) / "second-allocation.json")
            self.assertEqual((run_root / "t001-r1-control/campaign/binding.json").read_bytes(), before)

    def test_foreign_routing_is_refused_before_workspace_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, routing, output, run_root = self.make_inputs(directory)
            document = json.loads(routing.read_text())
            document["assignments"][0]["workspace"] = "/foreign/not-in-plan"
            write(routing, document)
            with self.assertRaisesRegex(ValueError, "foreign|omits"):
                prepare(plan, routing, "c550-a", output)
            self.assertFalse(run_root.exists())

    def test_changed_compiler_binding_is_refused_before_workspace_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, routing, output, run_root = self.make_inputs(directory)
            document = json.loads(plan.read_text())
            document["bindings"]["control"]["commit"] = "0" * 40
            write(plan, document)
            with self.assertRaises((ValueError, subprocess.SubprocessError)):
                prepare(plan, routing, "c550-a", output)
            self.assertFalse(run_root.exists())


class ImmediateHandle:
    def poll(self):
        return 0


class FakeLauncher:
    def __init__(self, state):
        self.state = state
        self.started = []

    def start(self, run, claim):
        self.assert_claimed(run["assignment_id"])
        self.started.append(run["assignment_id"])
        return ImmediateHandle()

    def assert_claimed(self, identity):
        if not (self.state / "claims" / f"{identity}.json").is_file():
            raise AssertionError("provider side effect preceded durable claim")

    def terminal(self, run, handle, exit_code):
        first = run["workload_ids"][0]
        return {"assignment_id": run["assignment_id"],
                "execution": {"status": "terminal"},
                "correctness": {"status": "passed"},
                "performance": {"status": "coverage_missing"},
                "confirmation": {"status": "missing"},
                "workloads": {first: {"correctness": {"status": "passed"}}}}


class FakeRelease:
    def observe(self, run, claim, terminal):
        return {"assignment_id": run["assignment_id"], "device_id": run["device_id"],
                "status": "unverified" if run["assignment_id"] == "a" else "verified"}


class FailingLauncher(FakeLauncher):
    def start(self, run, claim):
        self.assert_claimed(run["assignment_id"])
        self.started.append(run["assignment_id"])
        raise RuntimeError("provider outcome unknown")


class SequenceClock:
    def __init__(self, values):
        self.values = iter(values)
        self.last = 0

    def __call__(self):
        self.last = next(self.values, self.last)
        return self.last


class OwnerTests(unittest.TestCase):
    def allocation(self, root):
        runs = []
        for identity, device in (("a", "device-0"), ("b", "device-0"), ("c", "device-1")):
            run_root = root / f"run-{identity}"
            binding = {"assignment_id": identity, "task": "L1/fixture", "condition": "control",
                       "replicate": 1, "device_id": device, "author_capacity": 1,
                       "root": str(run_root.resolve()),
                       "workload_ids": [f"{identity}-workload-{index:02d}" for index in range(16)],
                       "author": {"wall_time_seconds": 100, "confirmation_seconds": 20}}
            commit = repository(run_root, {"campaign/binding.json": json.dumps(binding)})
            runs.append({**binding, "prepared_commit": commit})
        path = root / "allocation.json"
        write(path, {"schema_version": 1, "kind": "c550_bench_host_allocation",
                     "host": "c550-a", "start_window_seconds": 1000, "runs": runs})
        return path

    def test_unverified_release_blocks_same_device_and_preserves_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allocation = self.allocation(root)
            state = root / "state"
            launcher = FakeLauncher(state)
            result = CampaignOwner(allocation, state, sleeper=lambda _: None).run(launcher, FakeRelease())
            self.assertEqual(set(launcher.started), {"a", "c"})
            self.assertNotIn("b", launcher.started)
            self.assertEqual(result["assigned"], 3)
            rows = {row["assignment_id"]: row for row in result["rows"]}
            self.assertEqual(rows["a"]["release"]["status"], "unverified")
            self.assertEqual(rows["b"]["execution"]["status"], "unstarted_release_blocked")
            self.assertEqual(rows["c"]["release"]["status"], "verified")
            self.assertEqual(len(rows["b"]["workloads"]), 16)
            self.assertEqual(rows["a"]["workloads"][1]["correctness"]["status"], "unexamined")
            with self.assertRaises(FileExistsError):
                CampaignOwner(allocation, state, sleeper=lambda _: None).run(launcher, FakeRelease())

    def test_review_requires_terminal_and_verified_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allocation = self.allocation(root)
            state = root / "state"
            CampaignOwner(allocation, state, sleeper=lambda _: None).run(FakeLauncher(state), FakeRelease())
            with self.assertRaisesRegex(ValueError, "verified"):
                record_review(allocation, state, "a", {"assignment_id": "a", "status": "accepted"})
            record_review(allocation, state, "c", {"assignment_id": "c", "status": "unresolved"})
            self.assertEqual(report(allocation, state)["rows"][2]["review"]["status"], "unresolved")

    def test_confirmation_claim_is_single_use_and_inside_original_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            write(state / "claims/a.json", {"assignment_id": "a", "search_stop_at_epoch": 15,
                                             "stop_at_epoch": 20})
            candidate = {"id": "candidate-1", "status": "eligible", "completed_at_epoch": 14}
            claim = claim_confirmation(state, "a", candidate, now=16)
            self.assertTrue(claim["confirmation_opportunity_consumed"])
            with self.assertRaises(FileExistsError):
                claim_confirmation(state, "a", {**candidate, "id": "candidate-2"}, now=17)
            with self.assertRaises(ValueError):
                claim_confirmation(state, "a", {**candidate, "completed_at_epoch": 16}, now=17)

    def test_launch_failure_consumes_author_opportunity_and_blocks_same_device(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allocation = self.allocation(root)
            state = root / "state"
            launcher = FailingLauncher(state)
            result = CampaignOwner(allocation, state, sleeper=lambda _: None).run(launcher, FakeRelease())
            self.assertEqual(set(launcher.started), {"a", "c"})
            rows = {row["assignment_id"]: row for row in result["rows"]}
            self.assertEqual(rows["a"]["execution"]["status"], "unknown")
            self.assertEqual(rows["b"]["execution"]["status"], "unstarted_release_blocked")
            self.assertTrue(json.loads((state / "claims/a.json").read_text())["author_opportunity_consumed"])

    def test_start_window_expiry_keeps_every_unstarted_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allocation = self.allocation(root)
            state = root / "state"
            clock = SequenceClock([0, 2000, 2000, 2000])
            result = CampaignOwner(allocation, state, clock=clock, sleeper=lambda _: None).run(
                FakeLauncher(state), FakeRelease())
            self.assertEqual(result["assigned"], 3)
            self.assertTrue(all(row["execution"]["status"] == "unstarted_window_expired"
                                for row in result["rows"]))

    def test_assignment_identity_cannot_escape_state_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allocation = self.allocation(root)
            document = json.loads(allocation.read_text())
            document["runs"][0]["assignment_id"] = "../foreign"
            write(allocation, document)
            with self.assertRaisesRegex(ValueError, "identities"):
                CampaignOwner(allocation, root / "state")


if __name__ == "__main__":
    unittest.main()
