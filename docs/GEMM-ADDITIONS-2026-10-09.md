# Two standalone GEMM tasks

Suite version 2 appends the same two original tasks as
[bw1100-bench](https://github.com/qhy991/bw1100-bench/blob/main/docs/GEMM-ADDITIONS-2026-10-09.md).
The initial ten task records, source/data revisions, seed 200 and ten-round
correctness rule remain unchanged. The catalog now contains twelve tasks,
192 original workloads and 1920 correctness checks for a full suite.

| Task | Input/output precision | Dimensions | Coverage |
| --- | --- | --- | --- |
| L1/003 LM head projection | BF16 | M=batch×sequence, K=2048, N=102400 | M=128..8192, including irregular M |
| L1/077 Whisper output projection | FP16 | M=batch×sequence, K=1280, N=51866 | M=1..39232, decode and N tile tails |

Both compute `hidden_states @ weight.T` and return a tensor retaining batch and
sequence axes. Despite its title, all sixteen original L1/003 workloads have
`logits_to_keep == seq_len`; the input ABI contains no slicing scalar. Keep the
complete output projection. No shapes or tolerances are changed for C550.

These tasks isolate matrix multiplication from the existing fused GateUp task.
They let later experiments examine tiling, execution-group selection, memory
access, small-M overhead and tail handling under two precisions.

## Preparation and evidence

`sources.lock.json` continues to own the original source and dataset revisions.
Raw definitions, references and workloads stay in ignored `.data/`. Version 2
writes `.data/materialization-suite-v2.json`, preserving an existing version-1
materialization receipt. Changed raw files are still refused rather than replaced.

`examples/l1_003_torch_lm_head.py` and `examples/l1_077_torch_whisper_output.py`
are independent MetaX PyTorch vendor-GEMM candidate interfaces. Their arithmetic
uses PyTorch, and they are not CAKE optimization results. A C550 baseline requires
the original 16 workloads × 10 rounds to pass; performance needs a separate,
qualified measurement contract. This addition supplies no device result or speedup.

The largest original outputs are approximately 1.56 GiB and 3.79 GiB respectively.
Account for reference copies, candidate outputs, temporary storage and any
retained timing cohort before device evaluation. An input-size estimate alone
does not establish that a performance harness fits in memory.

Historical ten-task result collections retain their original scope. The new
catalog applies to a successor preparation and evaluation, not to old reports.

## Local software checks

Four regression tests passed: the C550 task catalog, extension while preserving
the version-1 receipt, refusal of changed reference data, and refusal of duplicate
workload IDs. The old ten task records, runner and source lock were compared with
the predecessor and are unchanged.

Four small synthetic CPU interface controls passed for the two Torch candidates,
covering batch/sequence output axes, BF16/FP16 dtype and unchanged input values.
These controls do not use original GEMM workloads. Default Python initially lacked
Torch; the controls used an existing Torch environment without changing it.
Original-workload materialization/audit and C550 device qualification remain pending.
