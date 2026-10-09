# c550-bench

面向 **MetaX C550** 的固定 SOL-ExecBench 正确性工作集。与
[qhy991/bw1100-bench](https://github.com/qhy991/bw1100-bench) 共用同一上游源码、数据修订、
12 题、192 个原始 workload、每 workload 10 轮新输入及原始比较器；只改变设备身份与
MetaX 的实际执行入口。可用于独立 Torch、Triton-MetaX 或其他候选的 return-value ABI
验证，**不是性能基准或 NVIDIA 官方分数**。

## 题目与来源

`suite.json` 是选题和预算的唯一来源；`sources.lock.json` 固定
SOL-ExecBench 源码和数据版本。难度 1–5 是人工估计，不是测量排名。

| 题目 ID | 实际内容 | 难度 | 覆盖的挑战 |
|---|---|---:|---|
| L1/069 | Residual + RMSNorm | 1 | 内存带宽、归约、BF16 舍入 |
| L1/011 | Llama3 RoPE 频率缩放 | 2 | 整数位置、分段缩放、三角函数 |
| L1/048 | 双投影 + GELU-tanh 门控 | 3 | GEMM 与 epilogue 融合；题名虽含 swiglu，参考实现不是 SiLU |
| L1/058 | 专家稳定排序与前缀和 | 3 | 整数精确性、稳定顺序、histogram/scan |
| L1/001 | GQA attention backward | 4 | softmax/dropout 反向、跨组归约、两个梯度输出 |
| L2/035 | ConvNeXtV2 + GRN | 3 | FP32 视觉负载、深度卷积、布局转换、多种归约 |
| L2/018 | 变长视觉 attention | 4 | cu_seqlens、head_dim=72、RoPE 和投影 |
| L2/024 | 256 专家 MoE dispatch/compute/combine | 4 | top-8、不规则 gather/scatter、grouped GEMM、大权重 |
| L2/060 | Chunk gated delta-rule attention | 5 | chunk=64、尾部 padding、三角更新、递归状态 |
| L2/056 | 完整 decoder layer backward | 5 | 十个梯度输出、混合 dtype、attention/MLP/norm 反向组合 |
| L1/003 | BF16 LM head GEMM | 2 | K=2048、N=102400，大词表规则 N 和不规则 M |
| L1/077 | FP16 Whisper output GEMM | 3 | K=1280、N=51866，M=1 decode、N 尾块和大 M |

Suite v2 新增两个独立 GEMM，每题保留 16 个原始 workload；全套预期共 1920 次正确性检查。
形状、ABI 和资格边界见 [GEMM additions](docs/GEMM-ADDITIONS-2026-10-09.md)。
新题的 Torch 示例尚未取得 C550 设备资格或性能结果。

完整保留所有原始 shape、dtype、容差和 UUID，不为了适应显存而缩小。
L2/024 单输入约 12 GiB，还需要 reference、候选与中间结果；应单独安排设备。
原始题库有 NVIDIA Evaluation Dataset License，raw Parquet、reference、生成的
workload 留在 Git 忽略的 `.data/`，不会上传；许可证边界见
[THIRD_PARTY.md](THIRD_PARTY.md)。

## 准备和 CPU 验证

准备环境建议 Python 3.12，不修改厂商 MetaX PyTorch：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-prepare.txt
.venv/bin/python scripts/prepare.py
.venv/bin/python c550bench.py audit --output .local/audit.json
uv pip install --python .venv/bin/python -r requirements-cpu-test.txt
.venv/bin/python -m unittest discover -s tests -v
```

`prepare.py` 固定上游修订，并复用其 `gen_inputs`、reference 和
`compute_error_stats`；已有准备目录若与锁文件不符则拒绝覆盖。
扩展后的准备回执写入 `.data/materialization-suite-v2.json`，保留已有十题的
`.data/materialization.json`，可复用原始 Parquet 缓存继续准备新增两题。
`python3 c550bench.py list` 只依赖标准库。CPU 自检、audit 和编译不会升格为
C550 设备资格，实际结果记录在 [验证记录](docs/VALIDATION.md)。

## C550 设备验证

先核对被授权的 C550 容器、GPU 占用和源码/数据锁；再把此目录（含忽略的
`.deps/`、`.data/`）放入该容器可见路径。使用容器里现成的 MetaX PyTorch
（已观测到 `torch.version.maca` 和 `torch.cuda` 兼容入口）。本 runner 同时验证
`torch.version.maca` 非空、设备名恰为 `MetaX C550`。`torch.cuda` 报告的
`major=8,minor=0` 是兼容层属性，**不能**据此说硬件是 NVIDIA SM80；宿主物理
目标与编译家族需另行核对，见 [环境记录](docs/C550-ENV.md)。

候选 Python 文件导出 `run(*inputs)`，输入顺序与输出名/顺序沿用上游
`definition.json`。示例候选在
[examples/l1_069_torch_baseline.py](examples/l1_069_torch_baseline.py)。

```bash
# 部分设备探针：原始 smoke workload × 2 轮，不能获得 full_device_correctness
python c550bench.py check --task L1/069_rms_norm --device cuda:0 \
  --candidate examples/l1_069_torch_baseline.py --workloads smoke --rounds 2 \
  --output results/c550-rmsnorm-smoke.json

# 完整单题门：16 原始 workload × 10 轮，每轮重新生成输入
python c550bench.py check --task L1/069_rms_norm --device cuda:0 \
  --candidate examples/l1_069_torch_baseline.py \
  --output results/c550-rmsnorm-full.json
```

输出路径必须新建，首个输入、reference、candidate 或比较错误立即写失败报告。
仅当**独立候选**在 MetaX C550 上完成该题全部 16 个原始 workload 与 10 轮，
并逐轮通过，报告才设置 `full_device_correctness=true`。单题通过不等于整套题集通过。
每轮同时检查 shape/dtype、上游容差以及整数/bool 精确相等；详见
[正确性规则](docs/CORRECTNESS.md)。本脚本不分配 GPU，不隔离其他作业，运行者须
先确认设备归属；不触碰 c550-1 的部署服务。

## 当前边界

2026-10-01 的设备验证：10/10 题、160/160 原始 workload 完成定义/容差审计；
11 项软件测试通过；L1/069 的独立 Torch 候选完成 16 workload × 10 轮，
160/160 通过且最大绝对误差 0，只授予该单题的 `full_device_correctness=true`；
L2/060 原始 smoke 参考自检 2/2 通过，但不授予候选资格。原始收据哈希见
[验证记录](docs/VALIDATION.md)。其它八题设备结果仍未知。没有延迟、吞吐
或 speedup 结果；单题正确性不能推广到完整模型服务。
