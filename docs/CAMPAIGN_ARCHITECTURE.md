# MetaX whole-task campaign architecture

The Bench owns one finite campaign and one complete callable per assignment. The
implementation includes the author, trusted supervisor, device assay and release
observer. `campaign/protocol.json` keeps live execution unqualified until C550-2
qualification covers those exact mechanisms.

## Public commands

Open-Cake freezes the comparison through `tools/evolve.py freeze-bench`. The Bench
prepares its allocations without provider or GPU access:

```bash
python3 -m campaign prepare --plan /intent/plan.json --routing /intent/routing.json \
  --host c550-2 --output /intent/c550-2/allocation.json
python3 -m campaign report --allocation /intent/c550-2/allocation.json
python3 -m campaign serve --allocation /intent/c550-2/allocation.json --state /evidence/owner
python3 -m campaign reconcile --allocation /intent/c550-2/allocation.json --state /evidence/owner
```

`serve` reads the tracked protocol from its own clean, read-only source checkout.
It refuses before intake while timing, profile, isolation or release is unqualified.
It accepts no shell command, model override, deadline override or qualification file.
`reconcile` only seals retained terminal and release evidence after the owner and its
supervisors have exited. It never repeats an author, request or confirmation.

## Ownership and custody

`campaign/owner.py` owns dispatch claims, total deadlines, finite scheduling, terminals,
release publication and the full denominator. Each claim includes the prepared commit.
One assignment contains all sixteen original workload UUIDs and ten fresh correctness
rounds. The confirmation reserve is part of that assignment's total budget.

`campaign/live.py` implements `AuthorLauncher` and `ReleaseVerifier`. It starts one
trusted supervisor per assignment from read-only source under isolated Python imports.
The supervisor is bound to the owner's process lifetime. The launcher enforces the total
deadline if the supervisor stops making progress.

`campaign/supervisor.py` owns that assignment's author, requests and candidate custody.
It checks the frozen Compiler catalog before invoking the real HMZ asynchronous flow in
`campaign/author_flow.py`. One fresh HMZ session receives the selected scaffold and
`author_context(root)`. There is no second author or replacement after failure.

The author runs in a private bubblewrap PID and mount namespace. It sees the selected
Bench, Compiler and original task material read-only. Its candidate, Schedule, Compiler
action, request and private home directories are writable. The author has no GPU device
nodes or owner state. Assays use separate private namespaces with the exact selected
render node and canonical MACA lock files. Process groups and private namespaces bound
all child processes to their original deadline.

`campaign/custody.py` admits create-only requests. A request is claimed before source
copy or execution. Search snapshots every Python and JSON file in the candidate's own
directory, refusing links and nonregular files. A candidate id can be snapshotted once.
The saved directory is read-only in every assay. A profile or confirmation refers to
that saved directory, never the author's subsequently edited copy. The owner retains
unknown and incomplete attempts and closes search after an uncertain request outcome.

One eligible search result and matching profile determine the nominee. The supervisor
consumes `claim_confirmation` before confirmation starts. Failure retains that nominee
and cannot select another candidate or establish a new deadline.

## Whole-call assay

`campaign/assay.py` calls the unchanged `c550bench.check_problem` for the candidate and
fixed baseline. It uses all original UUIDs, ten fresh input rounds and the original
reference and comparator. The selected Compiler is imported in a fresh process for each
assay. Before allocation, the assay checks Python and package versions against the selected
Compiler's `runtime/hosts/xcore1002.json`. It uses that capture's build environment and
refuses a mismatch without switching images or patching the capture. Its canonical `local_broker.admit_local_job` and `observe_local_metax` own MACA
allocation, exact target admission and physical/runtime/PCI mapping.

Timing covers the complete `run(*inputs)` invocation. Each sample clones fresh inputs,
fills four times the Target's declared L2 size and synchronizes before the start event.
The start event completes before the callable begins. The entire device completes after
the callable returns and before the end event. Thus asynchronous work and host submission
and synchronization gaps belong to the interval. This is not a kernel-only duration.
The implementation uses eleven warmup calls and ten alternating baseline/candidate pairs
per UUID. It retains every sample, median and relative median absolute deviation.
A relative deviation above 0.10 fails measurement quality. No timing retry is automatic.

A separate profile invocation uses the selected Compiler's MCPTI activity collector.
It retains full-call activity for the baseline and the exact candidate snapshot on every
UUID. Missing records, invalid device activity and unmodeled activity kinds prevent
profile eligibility. The present collector models kernels and API calls. Copy and
memset kinds without decoded records report `coverage_missing`. Event duration never
substitutes for missing profile coverage.

## Release and recovery

The supervisor writes no campaign-level release certificate. After it exits,
`CanonicalRelease` checks every recorded owned process identity using PID and start
ticks. A fresh process imports the selected Compiler's canonical MACA lock primitives,
checks the existing legacy and physical-device lock identities, and acquires the shared
legacy lock and exclusive device lock. It retains that observation before closing its
own descriptors. Missing process custody, an alive owned process, a busy lock or a changed
lock returns `unverified`. The owner blocks later assignments on that physical identity.
It never kills a foreign process or releases another assignment's descriptors.

The allocation remains the report denominator. Unstarted, unknown, incorrect, failed
quality, missing profile and failed confirmation results all retain sixteen workload
rows. Correctness, measurement, profile, confirmation, release and independent review
are separate fields. No report automatically accepts a Compiler or claims a speedup.

## Remaining live qualifications

The committed protocol remains `live_execution=false`. CPU software tests establish
request and lifecycle behavior, not MACA timing or isolation qualification. The exact
immutable toolchain image, C550-2 physical mappings, complete asynchronous callable event
interval, cache reset, sample quality, MCPTI coverage, isolated HMZ author and real
canonical release require retained device evidence before a reviewed source change can
open intake. Full twelve-task material and fixed baseline admission are separate gates.
The observed historical ten-task material cannot establish those gates.
