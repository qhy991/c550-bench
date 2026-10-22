# Author scaffold: v5-restricted (C500 engineering bench)

- Author tool: Claude Code via run-private `--restricted` wrapper
- Event contract: v5 (path writes outside workdir rejected before file creation)
- Model: glm-5.3:high
- Wall budget: 3 hours total; 1080 s confirmation reserve
- Reference access: oracle only (correctness check); reference implementation not provided to author
- Allowed inputs: task definition, workload axes, ABI spec, prior results from open-cake-ir
- Known constraints: MACA/xcore1000 target, MetaX C500 device
