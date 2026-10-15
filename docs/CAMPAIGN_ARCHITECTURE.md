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
The supervisor is bound to the owner's process lifetime. An independent guardian kills the owned process group at the execution cutoff even
when its controller blocks. The cutoff reserves fifteen seconds of the original total
budget for process exit and canonical release. Cleanup adds no grace period.

`campaign/supervisor.py` owns that assignment's author, requests and candidate custody.
It checks the frozen Compiler catalog before invoking the real HMZ asynchronous flow in
`campaign/author_flow.py`. One fresh HMZ session receives the selected scaffold and
`author_context(root)`. There is no second author or replacement after failure.

The author runs in a private bubblewrap PID and mount namespace. It sees the selected
Bench, Compiler and original task material read-only. Its candidate, Schedule, Compiler
action, request and private home directories are writable. The author has no GPU device
nodes or owner state. Assays use separate private namespaces with the exact selected
render node and canonical MACA lock files. The trusted assay keeps the canonical locks
open through candidate termination and its own device-context exit. Candidate workers
receive no lock descriptors. Their private PID namespace contains every descendant,
including descendants that create a new session.

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

`campaign/assay.py` retains `c550bench.check_problem` as the correctness owner. It
passes a candidate callable that performs a value exchange with a separate process.
The original input generator, reference, comparator, UUIDs and ten fresh rounds remain
unchanged. A persistent worker preserves candidate state across calls and executes the
complete `run(*inputs)` ABI.

Only the trusted assay imports the oracle and publishes correctness and result files.
The worker mounts its sealed bundle, two transport files, the selected immutable Compiler
and runtime dependencies. It cannot see the prepared author workspace, raw oracle,
trusted request, result directory or other processes. Standard input and output carry
bounded JSON metadata and tensor bytes. The decoder admits fixed dtypes and value kinds,
checks lengths before allocation, and never loads pickle or shared tensor storage.
Candidate output shapes and dtypes remain subject to the original comparator. A worker
exit or malformed response consumes the request and closes search as unknown.

Before allocation, the assay checks Python and package versions against the selected
Compiler's `runtime/hosts/xcore1002.json`. It uses that capture's build environment.
The canonical `local_broker.admit_local_job` and `observe_local_metax` own MACA allocation,
exact target admission and physical/runtime/PCI mapping.

Whole-call timing and profile currently return `coverage_missing`. The worker cannot
publish elapsed times, profile records or acceptance fields. Events recorded in the
trusted process do not establish completion or elapsed time in a candidate's context.
A native MCPTI collector inside the candidate process would still share candidate memory
and descriptors. Neither mechanism is used as trusted measurement.

The retained protocol specifies fresh clones, a four-L2 reset, eleven warmups, ten
alternating baseline/candidate pairs per UUID and the relative MAD gate. An independently
controlled C550 mechanism must establish that full interval, asynchronous completion and
separate MCPTI coverage for the exact saved candidate and fixed baseline before timing
can run. Host wall time and single-kernel records do not satisfy that contract.

## Release and recovery

The supervisor writes no campaign-level release certificate. After it exits,
`CanonicalRelease` checks every recorded owned process identity using PID and start
ticks. A fresh process imports the selected Compiler's canonical MACA lock primitives,
checks the existing legacy and physical-device lock identities, and acquires the shared
legacy lock and exclusive device lock. It retains that observation before closing its
own descriptors. Missing process custody, an alive owned process, a busy lock or a changed
lock returns `unverified`. The owner blocks later assignments on that physical identity.
It never kills a foreign process or releases another assignment's descriptors.

The committed frozen plan and routing remain the denominator. Preparation records a
`plan_source` commit reference. Admission, lifecycle ownership and reporting reconcile
every assignment id, task, condition, replicate, Compiler and physical route against it.
A missing arm is refused with its missing assignment ids before any launch or report.
The host allocation cannot redefine the denominator by dropping rows. Unstarted, unknown, incorrect, failed
quality, missing profile and failed confirmation results all retain sixteen workload
rows. Correctness, measurement, profile, confirmation, release and independent review
are separate fields. No report automatically accepts a Compiler or claims a speedup.

## Remaining live qualifications

The committed protocol remains `live_execution=false`. CPU software tests establish
request and lifecycle behavior, not MACA timing or isolation qualification. The exact
immutable toolchain image, C550-2 physical mappings, independent complete-call measurement,
cache reset, sample quality, MCPTI ownership, Linux namespace isolation, isolated HMZ author and real
canonical release require retained device evidence before a reviewed source change can
open intake. Full twelve-task material and fixed baseline admission are separate gates.
The observed historical ten-task material cannot establish those gates.
