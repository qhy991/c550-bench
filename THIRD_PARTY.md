# Sources and distribution boundary

- Evaluation framework: `qhy991/SOL-Execbench`, derived from NVIDIA/SOL-ExecBench
  revision `a9fa0804c793d438e70850c33fe34426e66d53dd`, Apache-2.0. The exact
  derivative revision is recorded in `sources.lock.json`. Its only source change
  sums the Boolean tolerance mask in pieces of at most `1 << 20` elements,
  accumulates integer counts globally, and converts the total to float once.
  Input generation, the reference, and the comparator's numerical decision are
  unchanged. The modules are loaded from the separately prepared checkout in
  `.deps/sol-execbench`. Reshaping a noncontiguous mask may copy Boolean storage,
  and the comparator still holds full-size arrays. This change establishes no
  GPU-memory repair or performance result.
- Dataset: `nvidia/SOL-ExecBench` on Hugging Face, NVIDIA Evaluation Dataset License.
  The pinned license URL is in `sources.lock.json`. Sections 1 and 3 distinguish
  internal evaluation/benchmarking from redistribution or hosting of the dataset.
  This repository does not include the dataset, references, input specifications,
  workloads, or Parquet files. `scripts/prepare.py` obtains them from their owner
  into ignored local storage and retains the original license there.
- Selection descriptions and difficulty estimates are this C550 project's editorial
  choices. IDs and URLs identify upstream resources; no measured performance or
  official NVIDIA leaderboard qualification is claimed.

Do not commit `.data/`, `.deps/`, materialized problems, raw input tensors, model
weights, or private machine inspection output. Publishing benchmark results does
not grant permission to redistribute the underlying dataset.
