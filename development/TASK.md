# Fixed-Compiler MetaX C500 evolve task: {task}

This is one task in the frozen 54-task C500 development denominator at Compiler
`{compiler}`. The exact Workload, starter, target `xcore1000`, numerical policy,
reference access, total deadline, confirmation reserve and measurement protocol are
fixed by `campaign/development-binding.json`.

## Required skills and tools

Read these files completely before changing a candidate, and record the reads in
`campaign/JOURNAL.md`:

- `$evolve`: `.deps/cake-ir/skills/evolve/SKILL.md`
- `$metax-c550-hardware-exploration`:
  `.deps/cake-ir/skills/metax-c550-hardware-exploration/SKILL.md`

The Run contains a frozen Compiler catalog in `campaign/compiler-api.json`. Every HMZ
turn receives that catalog and this Run's own action history. Use only:

- `python3 campaign/compiler_tools.py inspect --parent OWN_PATH`
- `python3 campaign/compiler_tools.py transform --request OWN_JSON --id UNIQUE_ID`
- `python3 campaign/compiler_tools.py verify --id UNIQUE_ID`

A transform result is a candidate, not a correctness or performance result. Keep the
request, parent, result and generated Program/stages. Never edit the catalog or reuse an
action id.

Evaluate an authored Cake candidate with a fresh id:

```bash
python3 -m development.runtime evaluate \
  --candidate campaign/candidates/CANDIDATE.py --id SEARCH_ID \
  --phase search --route cake
python3 -m development.runtime evaluate \
  --candidate campaign/candidates/CANDIDATE.py --id PROFILE_ID \
  --phase attribution --route cake
```

The owner will consider only correctness-qualified search evidence with a separate
accepted attribution result. Before the search cutoff, run one attribution on your best
accepted search candidate, even when it is the starter and no candidate beat it: without
it the owner has nothing to confirm, and a valid "the starter is best" result ends as
no qualified result. Do not invoke final confirmation.

## Default Cake route

The evaluation command submits a request to this task's controller and waits for its
response. The controller performs isolated compilation and device evaluation outside
the author's process sandbox. Keep the request, response and outcome. Do not invoke the
controller directly or change its environment. A controller fault closes search;
record the diagnosis and stop instead of repeating the same check in another session.

Start from `campaign/candidates/starter.py`. Author complete Cake Python Schedules under
`campaign/candidates/`. Preserve every input-dependent computation, public ABI, dtype,
rounding boundary, output ownership and original case. Evaluate only through the campaign
owner. The evaluator applies the external oracle before timing and retains an independent
profile. Do not call a GPU, compiler or baseline directly from authored code.

The Cake route also accepts complete Schedule or Program JSON in a `.json` file. To
evaluate a Compiler transformation, first run `compiler_tools.py verify --id ACTION_ID`,
then submit its retained `campaign/compiler-actions/ACTION_ID/program.json` or complete
`stage-NNN.json` with `--route cake`. Use the same search and attribution commands above
with that JSON path. Keep the action records and output path. Do not submit a tool's
`result.json` summary as a candidate. JSON uses the same Workload, target, ABI, semantic
and lowering checks; a multi-stage Program can still be refused by the selected target's
execution or measurement coverage. Never label such a refusal as a Python syntax error.

## Separate lower-level route when Cake cannot express the hypothesis

If a concrete optimization cannot be represented by Cake IR or cannot be realized by the
assigned lowering, retain the smallest Cake candidate and the exact assessment, lowering
or generated-source diagnosis first. A generic compile error is not proof of an IR gap.

When such a limitation blocks a concrete hypothesis, a native probe is required within
this task's remaining search budget; if the budget or infrastructure prevents it, retain
the specific reason and evidence instead. Create a **separate** native Triton engineering request under
`campaign/native-engineering/<id>/`. Its `plan.json` must name the blocked Cake candidate,
the specific missing commitment, inherited material, exact `xcore1000` target, and the
same Workload/oracle/numerical/measurement contract. The native candidate uses the
qualified kernel-only Triton JSON interface; it cannot call Torch or an existing
implementation. It shares this task's original 10,800-second deadline and 1,080-second
confirmation reserve.

Never switch routes inside an evaluated candidate. Never replace a failed Cake result with
a native result or report native timing as Cake timing. Final confirmation nominates only a Cake
candidate; native results remain engineering evidence beside it. Report both routes and the reason
for the split. A validated native mechanism can become a later Compiler/IR/lowering
proposal only through a successor commit, its analyses, counterexamples, Corpus Gate and
separate C500 qualification. `No promotion` is valid.

After writing the required `plan.json` and native candidate JSON, use the same evaluator
with `--route native_triton`. The candidate must remain below the plan directory. Use
fresh search and attribution ids and retain both Cake and native observations. Read
`campaign/native-interface.json` for the exact kernel entry, constants, options and grid;
do not infer or widen them.

## Budget and evidence

The task has 10,800 seconds from intake: 9,720 seconds for preparation, authoring, queue,
correctness, timing and profiling, then 1,080 seconds for the owner's one fixed
confirmation. GLM-5.3 high token use is accounting only. Read the deadline before every
evaluation. Do not reset a deadline, replace a terminal failure, retry an unknown attempt,
or alter the oracle, tolerance, target, evaluator, baseline or stopping rule.

Keep concise hypotheses and outcomes in `campaign/JOURNAL.md`. Record reusable mechanisms,
negative results, owning-layer leads and promotion disposition in `campaign/EVOLUTION.md`.

Every task is also a Compiler study. Before the search cutoff, append one line per distinct
Cake limitation you actually met to `campaign/gaps.jsonl` (one JSON object per line; create
the file if absent): `hypothesis`, `operation`, `limitation` (the exact diagnostic, refusal
code or generated-source/TTGIR/mcfatbin observation), `supported_transforms` (what you tried
and why it could not express the mechanism), `native` (the native request id and its
outcome, or the specific skip reason), `measured_effect` (observed timing or allocation
deltas, or "none"), `proposed_owner` (Verifier, cost model, IR vocabulary,
lowering, toolchain, host, Workload or baseline) and `evidence` (own `campaign/` paths).
A task that met no real limitation appends nothing. Candidate misuse, resource waits and
measurement noise are not limitations.

The campaign owner decides the terminal result and verifies device release.
