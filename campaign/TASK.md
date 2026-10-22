# Frozen whole-task C550 Bench search

Optimize only the task named by `campaign/binding.json`. The candidate is one Python
module exporting the task's complete `run(*inputs)` ABI and may dispatch across the
sixteen original workloads. The original definition, input generator, reference,
workload UUIDs, dtype, precision boundaries, tolerance and correctness rounds are fixed.

Read `campaign/binding.json`, `task-workloads.json`, `budget.json`, the selected Compiler
documents in `.deps/cake-ir`, and the frozen Compiler catalog. Use only this workspace's
candidate and action history. Do not read another assignment, condition, result or
candidate unless the frozen material policy explicitly names it.

The campaign owner controls the one total deadline, candidate snapshots, original
correctness assay, qualified event timing, profiling, fixed nomination, confirmation and
device release. An author does not certify a win, extend the clock, replace a failed
nominee or restart a claimed Run. Token use is accounting only.

To evaluate a candidate, run:

    python3 campaign/evaluate.py --candidate <path-to-candidate.py> --id <unique-lowercase-name>

This runs the full correctness gate (all 16 workloads × 10 rounds on MetaX C500) and
paired wall-clock timing. Results appear in `campaign/evaluations/<id>/outcome.json`.
Accepted outcomes include `conservative_geomean` (higher is better). For Cake IR
schedules, also pass `--cake-artifact <receipt-path>`.

Profiler coverage is not available on this platform; the skip reason is recorded
automatically. Correctness, performance, and release remain separate evidence gates.
