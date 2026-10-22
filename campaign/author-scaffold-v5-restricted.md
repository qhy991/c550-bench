# Author scaffold: v5-restricted (C500 engineering bench)

- Author tool: Claude Code via run-private `--restricted` wrapper (single-session; not hmz multi-round)
- Event enforcement: `--restricted` blocks path writes outside the workdir; no stream-parser terminal monitoring
- Model: glm-5.3:high
- Wall budget: 3 hours total; 1080 s confirmation reserve
- Reference access: see `campaign/binding.json` → `author.reference_access` for this Run's specific setting
- Allowed inputs: task definition, workload axes, ABI spec, Compiler documents in `.deps/cake-ir`
- Known constraints: MACA/xcore1000 target, MetaX C500 device
- Note: author condition differs from node4 (node4 uses hmz multi-round with budget.yaml and Lab v5 stream parsing)
