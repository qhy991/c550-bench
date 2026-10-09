# Source and scope

This project is a hardware adaptation of the user's private
`qhy991/bw1100-bench` at source commit
`f73bed29ec564e3bc68235f2d23111b53d822a5e`.
The ten task IDs, 160 workload UUIDs, seed, ten-round budget, original
reference/data revisions, CPU comparison and discrete-output rule were
retained. The executable is renamed `c550bench.py`; Hygon HIP/gfx938 guards,
DTK/Cake wrapper and BW1100 device receipts were removed. The device guard
now requires `torch.version.maca` and observed `MetaX C550` name; a reported
CUDA-compatible capability 8.0 is never used as NVIDIA architecture proof.
No SOL evaluator reward or timing path was transplanted.

Suite revision 2 follows the two GEMM additions published by
`qhy991/bw1100-bench`: `L1/003_lm_head_projection_with_logit_slicing` and
`L1/077_whisper_decoder_output_projection`. The preceding ten task records and
the pinned upstream source/dataset revisions remain unchanged. The extended
catalog contains twelve tasks and 192 original workloads; this catalog change
does not extend the historical ten-task device receipts.

`docs/VALIDATION.md` owns the bounded C550 device evidence. Individual raw
results and prepared upstream data stay out of Git under `.local/`, `results/`,
`.deps/`, and `.data/`. The fixed upstream source and dataset are bound by
`sources.lock.json`; see `THIRD_PARTY.md` for redistribution limits.
