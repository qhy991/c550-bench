# Frozen whole-task C550 Bench search

Optimize only the task named by `campaign/binding.json`. The candidate is one Python
module exporting the task's complete `run(*inputs)` ABI and may dispatch across the
sixteen original workloads. The original definition, input generator, reference,
workload UUIDs, dtype, precision boundaries, tolerance and correctness rounds are fixed.

Read `campaign/binding.json`, `task-workloads.json`, `budget.json`, the selected Compiler
documents in `.deps/cake-ir`, and the frozen Compiler catalog. Use only this workspace's
candidate and action history. Do not read another assignment, condition, result or
candidate unless the frozen material policy explicitly names it.

Read `.deps/cake-ir/docs/IR_GUIDE.md`, `.deps/cake-ir/compiler/AUTHORING_CONTRACT.md`,
and `.deps/cake-ir/compiler/targets/xcore1000.json` before authoring a Cake IR schedule.
Record each Read in `campaign/JOURNAL.md`. A readable path alone is not evidence of a
Read. Use `python3 campaign/cake_bridge.py emit <schedule> --output-dir <dir>` to
generate a receipt, and retain the receipt for every declared Cake artifact.

The campaign owner controls the one total deadline, candidate snapshots, original
correctness assay, qualified event timing, profiling, fixed nomination, confirmation and
device release. An author does not certify a win, extend the clock, replace a failed
nominee or restart a claimed Run. Token use is accounting only.

All counted Cake GPU computation must be unchanged Cake emission with declared emission
receipts. Author a Schedule or use public Compiler transforms, emit with
`campaign/cake_bridge.py`, and retain every declared receipt. Wrappers preserve the
public ABI and may use the source-bound community calls. Native Torch/Triton/HIP math
or torch.compile is not a counted Cake solution. Never edit the Compiler, evaluator,
protocol or final confirmation.

To evaluate a candidate, run:

    python3 campaign/evaluate.py --candidate <path-to-candidate.py> --id <unique-lowercase-name> --cake-artifact <receipt-path>

The `--cake-artifact` argument is required. It runs the full correctness gate (all 16
workloads × 10 rounds on MetaX C500), caller-ownership checks, and paired wall-clock
timing. Results appear in `campaign/evaluations/<id>/outcome.json`. Accepted outcomes
include `conservative_geomean` (higher is better). Pass one `--cake-artifact` per
emitted receipt.

Profiler coverage is not available on this platform; the skip reason is recorded
automatically. Correctness, performance, and release remain separate evidence gates.
