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

Live intake is refused until the source-bound protocol is qualified. The supervisor
checks the frozen Compiler catalog and includes `author_context(root)` in the actual
prompt. Correctness, performance, profiler coverage, release and independent
attribution remain separate.

Write each complete callable in `campaign/candidates/<candidate-id>/candidate.py`.
It must export `run(*inputs)` with the original return-value ABI. Python and JSON
helper files may live in that same directory. Use a new candidate id for every source
revision. The controller snapshots the complete directory on its first search request.

Submit one search request and then profile that same snapshot:

```bash
python3 -m campaign submit --id search-one --candidate one --phase search
python3 -m campaign submit --id profile-one --candidate one --phase profile
```

The response names the retained result. A timeout or unknown response consumes that
request. Do not resubmit it, replace its candidate directory, or claim confirmation.
Profile reads the controller's saved snapshot even if your writable files change.
The controller chooses one eligible nominee and performs one final confirmation under
the original total deadline. A failed confirmation has no replacement.

Use `python3 campaign/hmz_compiler_tools.py catalog`, `inspect`, `transform`, and
`verify` for the selected Compiler. The prepared catalog and `author_context(root)`
are included in this author session. Keep journals and scratch files in your private
home or own candidate directories. The original task material, Bench and Compiler are
read-only. You have no direct GPU device access. All device work uses the request API.
