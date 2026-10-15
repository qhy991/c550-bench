# Whole-task campaign design rationale

## Problem

C550 Bench owns twelve original tasks and the complete-task correctness ABI, but it has
no persistent whole-task author campaign. The existing engineering pilot splits one task
into sixteen short Runs and keeps experimental timing inside open-cake-ir. The new owner
must let one fresh author produce a cross-shape wrapper without transferring Bench oracle,
budget, confirmation or result ownership into the Compiler repository.

## Usage

The caller freezes a comparison with `tools/evolve.py freeze-bench`, provides one complete
host/device routing document, runs `python3 -m campaign prepare`, then reads
`python3 -m campaign report`. Preparation creates all fresh workspaces and frozen Compiler
catalogs with no provider or GPU calls. Live execution is deliberately refused until the
MetaX gateway and assay are independently qualified.

## Shape

The frozen allocation is the single source of truth. One local lifecycle owner publishes
irreversible dispatch and confirmation claims before side effects. This gives at-most-once
author opportunities without pretending that provider startup and filesystem writes form
a distributed transaction. Uncertainty consumes the opportunity and remains visible.

The implementation uses the existing flat `c550bench.py` correctness leaf and a sibling
`campaign` package. It does not migrate the correctness module or add another Run engine.
The public interface hides allocation validation, workspace preparation, crash recovery,
release blocking and denominator projection. Filesystem schemas stay private.

## Synthesis decision

Candidate A was selected as the base by the independent arena judge, scoring 38/40. It
had the strongest irreversible claim, deadline and crash contract. Candidate B supplied
the rule that independent review requires terminal execution plus verified release.
Candidate C supplied explicit evidence, coverage-limitation, missing and unverified
variants. The final design also requires all sixteen workload rows even after early
failure.

The design rejects Candidate C's caller-controlled reserve/record/finish API because it
splits launch and verdict ownership. It rejects Candidate B's ambiguous gap between
reservation and provider start. It also rejects a `c550bench.campaign` hierarchy because
`c550bench.py` is an existing flat module.

## Tradeoffs accepted

- An ambiguous dispatch cannot be retried, in exchange for never silently granting a
  second author budget.
- Preparation refuses an existing destination, in exchange for create-only custody and
  no hidden repair owner.
- One local owner and a filesystem with verified exclusive-create semantics are required,
  in exchange for avoiding a distributed queue service.
- The first change prepares, reports and tests scheduling policy but refuses live service,
  in exchange for not treating an unqualified generic command as a MetaX adapter.

## Alternatives considered

- Importing TaskLab/Ralph would move independent Bench lifecycle into open-cake-ir and
  preserve the per-case representation that this design must replace.
- Renaming the BW1100 scripts would leak HCU routing, DTK, ROCm release and wall timing.
- A resumable database queue would add leases and retry semantics without making provider
  or GPU side effects transactional.

## Implementation reconciliation

Issue #3 and PR #4 established the preparation, report and internal owner boundary.
Issue #7 adds concrete source-bound adapters behind the same owner. Two independent
design passes in the successor work selected a per-assignment trusted supervisor over
a persistent evaluation service. A persistent service would introduce another lifecycle
and recovery coordinator without a second consumer.

The public `serve` command exists and refuses before intake while the tracked protocol
is unqualified. `reconcile` only publishes terminal evidence after owner loss. Author,
assay and release use fixed source-selected entry points. A caller cannot provide a shell
command or qualification document.

Preparation clones the pinned upstream helper source and copies the assignment's original
task material into its ignored `.data` directory. The prepared data binding records the
copy and its source root. The author and assay see that copy read-only. Raw data remains
outside source control. Full suite material admission remains distinct from software tests.

The shared `hmz_compiler_tools.py` remains checked against its producing commit, copied
into each workspace and used to freeze the Compiler catalog. The concrete HMZ flow checks
that catalog and includes `author_context(root)` in the actual fresh author prompt.

The whole-call timer synchronizes a start event before `run(*inputs)` and waits for all
device work before its end event. It deliberately includes host submission and
synchronization gaps. The original `c550bench.check_problem` remains the correctness
owner. Profile and confirmation consume the same saved candidate directory.

## Qualification limits

The implementation leaves live intake closed. Actual isolated author operation, complete
callable timing with asynchronous work, MCPTI coverage and canonical release require
C550-2 evidence. The source-bound runtime paths and physical/PCI mappings are discovered
inputs, not qualifications. The selected Compiler host capture owns Python and package
versions. A different image must not be substituted to make that capture pass.

The existing MCPTI collector does not decode copy or memset records. A full-call profile
that observes those kinds reports a coverage limitation. It cannot make a candidate
eligible. Fixed baselines and all twelve tasks' material require independent admission
before a comparison. No complete-task or performance result is established here.
