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

The implementation keeps only `prepare` and `report` in the public CLI. `CampaignOwner`
implements and tests the queue policy internally, but no `serve` command exposes it until
the MetaX gateway and assay are qualified. This narrows the initial public surface while
preserving the designed lifecycle contract; Issue #3 and PR #4 accept that boundary.

Preparation records the existing materialized task-data root and requires its Suite v2
source receipt instead of copying raw licensed material into Git. A future live adapter
must mount that exact root read-only and recheck the binding. This keeps raw data outside
source while making its custody an explicit live qualification requirement.

The selected shared `hmz_compiler_tools.py` is checked against its producing commit,
copied into each fresh workspace and used to freeze the Compiler catalog before intake.
The complete 25-test suite passed at `34e944e` in the existing C550 no-network, no-GPU,
no-key CPU container. A future live adapter remains a separate reviewed change and must
update the usage and outcome contracts here before it becomes callable.

## Open questions and risks

- Which existing MetaX gateway will own Run-wide admission closure and canonical release?
- Which event timer, interval, reset and profiler evidence will qualify complete callable
  measurement?
- Which stable physical identity reconciles container-visible ordinals and PCI devices?

These questions block live launch, not CPU-only preparation and owner tests.

## Next implementation step

Implement frozen allocation admission, create-only preparation and complete-denominator
reporting, then add the internal owner with crash injection around dispatch, confirmation
and release.
