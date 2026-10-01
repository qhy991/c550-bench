# 正确性约定

原始定义和 workload 的权威版本由 `sources.lock.json` 指定；`reference.py` 是
`definition.json.reference` 的可阅读投影。加载前必须一致。所有原始 shape、dtype、
custom input factory、scalar 和 workload UUID 均保留，不生成伪造的官方 workload。

输入通过原版 `sol_execbench.core.bench.io.gen_inputs` 生成；custom factory 直接来自
对应 reference。每轮重新生成输入，默认 seed=200、每个 workload 10 轮。
参考程序收到输入副本，候选收到原输入，避免参考程序的原地修改影响候选。
跨 CPU/MetaX CUDA-compatible path 的 RNG 序列不保证逐位相同；同一轮候选与参考使用相同输入值。

数值比较直接调用固定版本 `compute_error_stats`：

- 非有限值按上游规则拒绝；只有 workload 明确允许时，双方相同位置的 `-inf` 可排除。
- 逐元素误差界为 `atol + rtol * abs(reference)`，再检查 required matched ratio。
- 保留 `max_error_cap`，防止比例门漏掉少量巨大误差。
- shape 和 dtype 必须与参考一致，且参考本身必须符合定义；每轮都检查。
- 字典输出必须包含全部且仅包含已声明输出。
- **额外的离散输出门**：整数和 bool 必须逐元素精确一致。上游会将整数转成 FP32
  后做比例误差，可能放过错误排序/索引。本项目对此更严格，不能把本项目 PASS/FAIL
  不加区分地写成官方评测结果。

## 上游字段差异

选中的 L2 workload 使用 `required_match_ratio`，固定版评测器的字段名是
`required_matched_ratio`。当前 Pydantic 模型忽略前者，生效默认值为 0.99。
本项目保持这个**实际生效行为**，不把字段静默重命名而放宽到 0.98。
`c550bench.py audit` 同时输出 raw tolerance、effective tolerance 与未识别字段。
未来若升级上游，应在新修订中重新审查，不能覆盖旧结果。

## 证据范围

`reference_selfcheck` 只验证生成器、reference 和评测入口在给定环境下可执行并自洽；
它没有独立验证 reference 算法，也没有验证优化实现。CPU selfcheck 不能证明 C550设备
支持。候选模式比较真实候选与原始 reference。

此入口面向协作开发，不提供上游完整 reward-hack 防护、恶意代码隔离或计时验证。
无需为本次正确性工作集复制 NVIDIA clock lock、CUPTI 或 SOL 评分链。
C550资源分配、候选编译和 profiler 应接在各自既有入口，不能把 PASS 推广为性能收益。
