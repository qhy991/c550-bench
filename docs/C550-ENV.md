# C550-2 MetaX environment identity (observed 2026-10-01)

- Host: c550-2, eight MetaX C550 devices, no GPU process at preflight.
- Existing owned container: `glm53-flame-trace-0925`, image tag
  `cr.metax-tech.com/public-init/vllm/vllm0.26.0-glm5next:0821`.
- Host `mx-smi`: MACA `3.5.3.18`, kernel driver `3.6.11`.
- Container Python `3.10.10`, PyTorch `2.10.0+metax3.8.0.4.c600u`,
  Pydantic `2.13.4`, Triton `3.6.0`.
- `torch.version.hip is None`; `torch.version.maca` is `3.8.0.4.c600u`.
  `torch.cuda.device_count()` is eight; `torch.cuda.get_device_name(0)` is
  `MetaX C550`. `get_device_properties(0)` reports compatibility major/minor
  `(8,0)` and about 65 GiB; this is not NVIDIA SM80. The host card identity
  and Triton compiler target require separate checks before kernel conclusions.

This is an environment/identity probe only. No SOL task ran in this observation.
The suite still requires pinned source/data preparation and create-only device
receipts before any C550 correctness claim. Do not reuse C550-1 evidence as
C550-2 qualification.
