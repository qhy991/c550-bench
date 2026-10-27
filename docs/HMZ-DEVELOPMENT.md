# MetaX C500 HMZ development round

The registered C500 (`xcore1000`, host `metax-infini`, four cards) development round is
separate from the independent Bench campaign. `development/tasks.json` owns its 61-task
denominator, the tasks with a retained C500 Run; `metax_fp8_gemm` and TinyGEMM are the
unsupported-target rows. It was first written for two C550 hosts; that layout is retired.

The author is hmz 0.2.1's fenced Claude Code. `scripts/install_hmz_maca.sh` builds the
run-private `~/.cake-c500-hmz`, whose fence admits `/dev/mxcd`; stock hmz does not. Runs
live under the home directory, because the fence leaves `/tmp` and `/var/tmp` writable.
`python3 -m development.queue --host --python <MACA python>` runs each task controller on
the host under the render group; the C550 container launcher remains for that layout. A generated plan and all workspaces remain outside
Git.

Freeze from the retained C500 Runs with `scripts/freeze_development.py --retained /var/tmp/cake-c500-bench`. The plan
pins the open-cake-ir Compiler, this launcher commit, GLM-5.3 high, both author skills,
three-hour budget, static host/device routing, `--authors-per-device` (1 to
`MAX_AUTHORS_PER_DEVICE`, 4) and each task's original Workload/starter material. The broker
still runs one device lease per card, so authors beyond the card's evaluation throughput
wait for the lease with their search clock running; a lease wait beyond 300 s returns
`no_receipt`. With more than one author per card the queue starts authors on the same card at
least `START_SPACING_SECONDS` (780 s) apart, so authors that use their full search do not reach
their single final confirmations together. A confirmation can still meet a sibling's search
evaluation on the card lease, and an author that ends search early confirms early. Choose the count from the previous round's card busy share and lease wait. `python3 -m development` creates fresh HMZ workspaces without a provider or GPU.

Each prepared `campaign/TASK.md` requires `$evolve` and
`$metax-c550-hardware-exploration`. `campaign/compiler_tools.py` is the shared pinned HMZ
adapter from open-cake-ir. Preparation runs `freeze` before intake and commits the
catalog. A live owner must append `author_context(root)` to every HMZ turn and expose the
same adapter's `inspect`, `transform` and `verify` commands.

Cake is the default route. A retained expressibility or lowering diagnosis may open a
separate native Triton engineering request inside the same task deadline. The native
route keeps the Workload, exact target, oracle, numerical and measurement contracts and
never replaces the Cake result. It needs its own source admission, device qualification
and search/attribution evidence; final confirmation nominates only a Cake candidate, so a
native result stays engineering evidence beside it. A generic tool failure is not an IR gap.

Preparation is not launch qualification. Before intake, the live successor must prove:

1. exact Compiler/launcher/task and skill custody;
2. HMZ initial and resumed author turns containing the frozen catalog context;
3. isolated CPU compilation and exact physical/runtime/PCI device admission;
4. candidate correctness, ten-sample MACA event timing, separate MCPTI profiling and
   canonical release for both Cake and any enabled native route;
5. one create-only deadline and one terminal record per assignment, with no retries.

The live controller and its evaluation children run through the absolute
`development/controller_entry.py` in the pinned launcher checkout, mounted read-only,
under Python's isolated import mode. Each task links this checkout at `.deps/launcher`.
The author workspace's public evaluator submits a create-only request; the controller
services it outside the author's inherited seccomp. Qualification must exercise a real
isolated author request through correctness, event timing, independent profiling,
same-candidate confirmation and release. Controller-direct probes alone do not cover
that path.

`python3 -m development.report --allocation HOST_ALLOCATION --state HOST_STATE` derives
one host's full denominator from retained evidence. The report distinguishes live owners,
unstarted tasks, terminal errors, correctness, timing quality and final confirmation.
`development.report.combine` refuses missing tasks, duplicates and mismatched source
bindings before producing the54-task view. Coverage does not imply success, and an
accepted search or profile is not an optimization result. The journal and evolution paths
remain evidence for later review; the reporter does not infer promotion from author text.
Reporting may use a successor tool without changing a frozen task's Compiler or launcher.
