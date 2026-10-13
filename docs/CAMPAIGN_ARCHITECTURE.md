# MetaX whole-task campaign architecture

This document defines the CPU-verifiable campaign owner for fresh C550 Bench search.
It does not qualify a live device route or start an experiment.

## Supported use

Open-Cake freezes the complete comparison intent before any workspace exists:

```bash
python3 tools/evolve.py freeze-bench \
  --draft /intent/draft.json \
  --bench /checkouts/c550-bench \
  --control /checkouts/cake-control \
  --successor /checkouts/cake-successor \
  --output /intent/plan.json
```

The Bench then validates one complete routing document and prepares a fresh workspace
for every frozen allocation. Preparation has no provider or GPU capability.

```bash
python3 -m campaign prepare \
  --plan /intent/plan.json \
  --routing /intent/routing.json \
  --host c550-a \
  --output /intent/c550-a/allocation.json

python3 -m campaign report --allocation /intent/c550-a/allocation.json
```

Live execution remains unavailable until a separate change binds a reviewed MetaX
author process, exact device gateway, event timer, profiler policy and release receipt.
No generic shell command or user-supplied qualification file enables execution.

## Invariants

- One assignment contains one original task, all sixteen ordered workload UUIDs and one
  total wall budget. Search uses `total - confirmation`; confirmation has no new clock.
- The frozen allocation is the report denominator. Missing files, an unstarted Run and
  malformed evidence never remove an assignment or workload.
- A create-only dispatch claim is written before provider or device side effects. The
  claim permanently consumes that author opportunity, including the crash gap before
  process start. Confirmation uses the same irreversible claim rule.
- One owner writes lifecycle records. Authors only propose source. The owner snapshots
  candidates and owns evaluation, nomination, confirmation and terminal publication.
- The queue schedules authors. The admitted MetaX gateway alone owns device admission
  and release. Missing or contradictory release evidence blocks later authors on the
  same physical identity; independent devices may continue.
- Independent attribution review requires terminal execution and verified canonical
  release. It cannot change a nominee, restart search or add measurements.
- Correctness, performance, profiler coverage, confirmation, release and review remain
  separate result fields. A command exit or correct output does not imply a speed result.

## Internal shape

The public surface is preparation and reporting. The queue owner is internal until the
concrete live adapters are qualified.

```python
def prepare(plan: Path, routing: Path, host: str, output: Path) -> HostAllocation:
    """Create every fresh host workspace; call no provider or GPU."""
    raise NotImplementedError

def report(allocation: Path, state_directory: Path | None = None) -> CampaignReport:
    """Return every assignment and every original workload, including missing states."""
    raise NotImplementedError

class CampaignOwner:
    def run(self, launcher: AuthorLauncher, release: ReleaseVerifier) -> CampaignReport:
        """Drain unclaimed allocations without ever restarting a claimed author."""
        raise NotImplementedError
```

`campaign/binding.py` owns frozen plan, routing and prepared-workspace admission.
`campaign/prepare.py` owns create-only workspace construction and the pinned Compiler
tool catalog. `campaign/owner.py` owns claims, deadlines, finite scheduling and release
blocking. `campaign/report.py` derives the full denominator from the allocation and
retained evidence. `campaign/__main__.py` is a thin boundary.

The Compiler seam is an exact copy of the selected shared
`tools/hmz_compiler_tools.py`. Preparation runs `freeze` before intake. A future live
adapter must call `catalog(root)` at intake, add `author_context(root)` to every actual
prompt and expose `inspect`, `transform` and `verify` in a fresh condition process.
The Bench does not import the open-cake-ir Run engine.

## State and recovery

Per assignment the projected lifecycle is:

`prepared -> claimed -> searching -> nominated -> confirming -> terminal`

Failure, timeout, unknown and no-gain are terminal execution classifications. Release
and owner review are orthogonal. A malformed or uncertain claim is unknown and remains
consumed. Owner recovery may continue unclaimed allocations inside the start window;
it never restarts a claimed author or confirmation. Cleanup and release observation may
continue after the budget, but provider, authoring and measurement work may not.

Every assignment report contains all sixteen workload UUIDs. Each UUID separately
states correctness, measurement and confirmation as evidence, coverage limitation,
missing or unexamined. Early failure therefore preserves the remaining denominator.

## Live qualification still required

Production execution must refuse until evidence binds the exact physical C550 identity,
Compiler target and route, immutable toolchain image, event timer and measured interval,
device-state reset, warmup/sample policy, profiler coverage, process deadline enforcement
and canonical release terminal. HCU, ROCm, CUDA, CUPTI and DCU wall-timing assumptions
are not defaults for this campaign.
