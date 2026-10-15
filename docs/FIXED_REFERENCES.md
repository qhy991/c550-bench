# Fixed reference qualification

The existing frozen plan selects one tracked `baseline_files` path per task. The
selection can name an existing self-contained module with `run(*inputs)`,
`baselines/inductor_strict.py`, or `baselines/inductor_partial.py`. These paths
select implementation candidates. No source selection establishes performance
strength or device correctness.

`AssignmentSupervisor` prepares the selected adapter in private owner state before
any author work. `campaign.reference.prepare_reference` checks the frozen Bench
source and the canonical task projection. The original reference remains in the
ignored task material. The private adapter copy is a projection of the selected
tracked file. It is never mounted into the candidate worker or author namespace.
The author mounts mask the baseline and example directories and every selected
`baseline_files` path in the canonical frozen plan. They also mask
its trusted compilation adapter, Python bytecode caches, and Git metadata in both the prepared root and the trusted Bench source tree.
The original high-level specification and oracle remain available to the author.

The supervisor consumes one separately bounded reference opportunity before
starting the author. It uses the existing trusted entry, process deadline,
canonical device admission and release owner. Preparation and lazy compilation
consume the assignment's original clock. This step does not extend the deadline
or create a second scheduler. A failure or unknown process outcome prevents
author launch and retains the failed opportunity. Candidate assays consume the
source-bound qualification receipt and do not repeat the reference's 160 cases.

`campaign.reference.qualify_reference` executes the selected complete callable
through `c550bench.check_problem`. The Suite owns the original workload UUIDs,
round count, seed and semantics. The comparator retains every output and exact
discrete equality. The strict adapter compiles the original module's complete
`run` function, including its helpers and supplied-state ABI. It uses
`fullgraph=True`, `dynamic=False`, the installed Inductor backend, and the
runtime's default specialization limits. Compiler errors propagate.
`suppress_errors=False` and `fail_on_recompile_limit_hit=True` prevent an error or
recompile-limit event from silently replacing the selected implementation with
eager execution. A strict call with no compiled graph refuses qualification.
The captured MetaX Torch source implements these controls in
`torch._dynamo.config` and `torch._dynamo.convert_frame`.

The partial adapter allows graph breaks in the unchanged original call. Its route
explicitly declares data-dependent Python control flow, host conversion, and
unsupported original operations as possible eager regions. Its receipt retains
Dynamo graph-break reasons and compiled graph source. It makes no uniform
compilation claim. A failed backend compile or exhausted specialization limit
still fails. The bounded qualification process retains compiler logs, debug
output and generated code in private evidence outside the checkout. Generated
code presence does not prove actual device dispatch.

The result records source, runtime observations, route and original correctness
separately. All reference measurement, independent profile and performance fields
remain `coverage_missing`. A correctness pass cannot make the protocol live,
certify a strong denominator, or award a speedup. The entire twelve-task set still
requires original C550 receipts, complete-call timing, A/A neutrality, native
dispatch evidence, independent profile and canonical release before formal
comparison.

The installed MACA SDK also compiles a host admission helper, even for a library
GEMM. `prepare_reference` now builds and loads `MacaUtils` into its private
`runtime-helper` artifact during CPU preparation in the captured SDK. The
artifact filesystem must permit executable mappings. The SDK owns its cache key
and native module format; no vendor source or generated binary enters Git.
The original `/tmp` policy remains noexec.

Before admission, `prepared_maca_helper` loads that artifact with the SDK's scoped
cache setting and build-refusal hook. The same hook stays active through actual
route observation, so a missing or unloadable helper cannot trigger a host build
after the lease. Admission records the consumed helper. The helper cache enters
only the trusted assay namespace. Its settings are restored afterward, and the
candidate worker receives no private baseline bundle.

This artifact prepares only the SDK host utility. Inductor graph compilation,
Triton kernel/launcher caches and arbitrary candidate code generation require
their own explicit executable preparation and device-dependent setup evidence.
They do not inherit qualification from the host helper. Timing, profile and
strong-reference status remain unqualified.

Trusted reference and assay attempts use one native-cache environment owner in
`campaign.processes.trusted_attempt_environment`. It creates executable cache
roots only below the current output for Inductor, Triton and PyTorch extensions.
The supervisor supplies that environment before spawn. The isolated entry also
applies the same helper before importing the reference or assay, so standalone
calls have the same storage policy. `/tmp` remains noexec and TMPDIR stays `/tmp`.
A noexec output or redirected cache root refuses before admission.

Admission seeds only the CPU-prepared SDK host module into that attempt's Triton
cache, preserving the SDK's own relative cache key. It retains the source and
copy paths in runtime-helper-seed.json. The read-only preparation artifact is
never used as a writable kernel cache, and previous kernels or reference source
are not copied. Fresh compiler workers receive explicit attempt cache variables
through the installed Inductor forwarding path. The admission-only build refusal
still ends after route observation. Later graph-dependent compile, load and
autotune work is bounded target setup and is not timed performance evidence.
Candidate-worker caches remain private and separate. This trusted-cache change
adds no candidate JIT route, timer, profiler gate, allocator or execution mode.
