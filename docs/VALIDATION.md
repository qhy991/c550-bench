# C550 验证记录：2026-10-01

C550-2 自有容器 `glm53-flame-trace-0925`，MetaX vLLM 镜像 tag `0821`，
Python 3.10.10，PyTorch `2.10.0+metax3.8.0.4.c600u`，Pydantic 2.13.4，
Triton 3.6.0；八卡无进程的快照先于 GPU run，结束后亦无 GPU 进程。
部署副本位于宿主 `/root/experiments/glm53-flame-chase-20260925/c550-bench-20261001/`
（容器内 `/root/exp/c550-bench-20261001/`）；它是当时已授权的自有容器挂载目录，
不是仓库代码或长期存储。结果路径均为 create-only。
本仓库 runner SHA-256 `f42068107e18b7e2dede942320dbb14d2fb4d86fd4bb127597c3c24411ddfb4d`，
`suite.json` SHA-256 `d237bb82c53b1ee136efb6922e2aafc0139feb7aa582d6c87dbd21233064b076`，
`sources.lock.json` SHA-256 `650bb88a2a4fb8136b209c1f04b8ff30f1bb5ffd1ba6d0746a086a7d2c0d31cd`。
本地与容器副本哈希相同；固定上游源码 checkout 为
`a9fa0804c793d438e70850c33fe34426e66d53dd`。原始题目、参考实现和结果
位于 Git 忽略目录，遵守 [THIRD_PARTY.md](../THIRD_PARTY.md)。

| 检查 | 实际结果 | 证据范围 |
| --- | --- | --- |
| `python -m unittest discover -s tests -v` | 11/11 PASS，包括 MetaX/C550 身份拒绝门 | 软件契约；没有单独设备正确性结论 |
| `c550bench.py audit` | 10/10 题、160/160 原始 workload，保留 raw/effective 容差 | 报告 `.local/audit-20261001.json`，SHA-256 `7ceb69ce0ec73bd3276080af6e26b8119256a755b94f579d750558967953231f`；不跑候选 |
| L1/069 原始 smoke、独立 Torch 候选、2 轮 | 2/2 PASS，最大绝对误差 0，`full_device_correctness=false` | `results/c550-rmsnorm-smoke-20261001.json`，SHA-256 `7a1c8a9ca5191d47992ab1a903cf4b5107ca3854a5a9e2c693cd5a09e03e18aa`；部分设备探针 |
| L1/069 所有原始 workload、独立 Torch 候选、每项 10 轮 | 16×10=160/160 PASS；全部 case 通过；最大绝对误差 0；该题 `full_device_correctness=true` | `results/c550-rmsnorm-full-20261001.json`，SHA-256 `8d05c28f5f438a95660b6f06568af17a3cb7e61b67fb809a528f0d906aeff101`；仅本题全量正确性 |
| L2/060 原始 smoke、参考程序自检、2 轮 | 2/2 PASS，`full_device_correctness=false` | `results/c550-kda-reference-smoke-20261001.json`，SHA-256 `dc93a1267892775efdb1105c80a287d26b2e4083c0907fef8d6724e4d7bb9012`；未验证独立 KDA 候选 |

L1/069 最大原始输入约 268,451,840 B。L2/024 一份原始输入约 12 GiB，
其它八题尚未完成 C550 独立候选全量门。此套件是 correctness-only；没有计时、
官方 NVIDIA SOL 分数、AITER/Triton 性能结论或模型级收益。
