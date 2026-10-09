# c550-bench operating rules

- `suite.json` owns the selected tasks, original smoke UUIDs, seed and ten-round correctness contract. Suite version 2 contains twelve tasks. `sources.lock.json` owns upstream revisions. Preserve all original workload shapes and effective tolerances.
- Reuse pinned SOL-ExecBench input generator/reference/comparator. Candidate code may change, oracle semantics may not. Discrete outputs require exact equality in addition to upstream numeric rules.
- Raw dataset and generated definitions stay in ignored `.data/`, source checkout in `.deps/`. Never commit or publish raw material, even to a private repository.
- `list`, CPU audit/selfcheck, partial smoke and full C550 candidate checks have separate meanings. Only an independent candidate on the MetaX C550 device across all 16 workloads × 10 fresh rounds for its one task may set `full_device_correctness=true`. No latency or score is inferred.
- Check host ownership/idle state before device work; this CLI does not reserve GPUs. Keep c550-1 and foreign jobs untouched.
- Use create-only result paths and preserve failed receipts. Do not relabel MetaX CUDA compatibility `8.0` as physical SM80 or confuse C550 xcore identity with the Triton target family.
- Implement first, then run focused CPU software tests and original-workload audit. Device conclusions require actual C550 receipts.
