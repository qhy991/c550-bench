# Sources and distribution boundary

- Evaluation framework: NVIDIA/SOL-ExecBench, Apache-2.0. `sources.lock.json`
  pins `qhy991/SOL-Execbench` revision `0e08129caf4db959a6263e802e4a15f29a83ef32`,
  a derivative of official NVIDIA revision `a9fa0804c793d438e70850c33fe34426e66d53dd`.
  Its only source change replaces Boolean-mask `.sum()` with `torch.count_nonzero`
  in `compute_error_stats`. The input-generation module is unmodified. Both modules
  are loaded from the separately prepared checkout in `.deps/sol-execbench`.
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
