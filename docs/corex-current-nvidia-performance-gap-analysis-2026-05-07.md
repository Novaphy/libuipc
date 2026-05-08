# CoreX 当前默认路径与 NVIDIA 路径性能差距分析

日期：2026-05-07

## 结论摘要

当前 CoreX 默认最好路径已经从早期数百秒级优化到 `wb400 = 143.956s`，但与原生 NVIDIA 路径 `45.61s` 仍有约 `3.16x` 总耗时差距。

这个差距可以拆成两层：

1. **CoreX-compatible 算法与原生 NVIDIA 算法的差距**：在 NVIDIA 硬件上运行 CoreX-compatible float 路径为 `102.01s`，原生 NVIDIA 路径为 `45.61s`，约 `2.24x`。这部分几乎完全对应 PCG 总迭代数差距：`88085 / 38066 = 2.31x`。
2. **同一 CoreX-compatible 路径在 CoreX 硬件/运行时上的执行差距**：当前 CoreX `143.956s` 对 NVIDIA 硬件 CoreX-compatible `102.01s`，约 `1.41x`。两者 PCG 总迭代几乎相同：`88441 / 88085 = 1.004x`，说明这部分主要是硬件、编译器、runtime、kernel 和同步执行成本差距。

因此，当前剩余差距的主因不是 Newton 次数，也不是某一个还没迁移的 CPU fallback，而是：

- CoreX-compatible float 路径的线性系统/预条件/接触矩阵质量导致 PCG 每次 solve 平均迭代仍约 `36`，而原生 NVIDIA 是约 `14.4`。
- 同一 CoreX-compatible workload 在 CoreX 上执行比 NVIDIA 上慢约 `1.4x`。
- 部分非 PCG 热点仍在 contact、DyTopo/matrix conversion、SpMV、dotnorm/sync 等阶段，但它们应服务于前两项主线，而不是通过大范围开关 sweep 盲目尝试。

## 对照数据

### 三个关键运行结果

| 路径 | Wall | Newton sum | PCG calls | PCG iter sum | PCG iter max | 说明 |
|---|---:|---:|---:|---:|---:|---|
| 原生 NVIDIA | `45.61s` | `2248` | `2647` | `38066` | `83` | `SUMMARY.md` |
| NVIDIA 硬件 + CoreX-compatible float | `102.01s` | `1978` | `2476` | `88085` | `122` | `SUMMARY_5070ticorex.md` |
| CoreX 当前默认最好路径 | `143.956s` | `1959` | `2457` | `88441` | `122` | 2026-05-07 默认复跑 |

### 比例拆解

| 对比 | Wall ratio | PCG iter ratio | 解释 |
|---|---:|---:|---|
| CoreX 当前 / 原生 NVIDIA | `3.156x` | `2.323x` | 总差距 |
| NVIDIA CoreX-compatible / 原生 NVIDIA | `2.237x` | `2.314x` | 算法/数值路径差距 |
| CoreX 当前 / NVIDIA CoreX-compatible | `1.411x` | `1.004x` | 同 workload 的硬件/runtime 差距 |

一个重要观察是：CoreX 当前默认路径的 Newton sum 和 PCG calls 都不比 NVIDIA 原生路径高，甚至略低：

```text
Newton sum ratio = 1959 / 2248 = 0.871
PCG calls ratio  = 2457 / 2647 = 0.928
```

但是单次 PCG 的平均迭代数显著更高：

```text
原生 NVIDIA             = 38066 / 2647 = 14.38
NVIDIA CoreX-compatible = 88085 / 2476 = 35.58
CoreX 当前默认          = 88441 / 2457 = 36.00
```

这说明剩余主要算法差距集中在每个线性系统的 PCG 收敛质量，而不是外层 Newton/line-search 调用次数。

## 差距 1：PCG 迭代数仍是最大算法差距

5 月 6 日的 `12x12` ABD block-inverse preconditioner 已经把 CoreX 的 `wb400` PCG iter sum 从 `168717` 降到约 `88478`，这是当前最有效的一轮优化。它恢复了 NVIDIA 路径使用完整 ABD block inverse 的算法形态，明显强于旧 Jacobi preconditioner。

但与原生 NVIDIA 的 `38066` 仍有：

```text
88441 / 38066 = 2.323x
```

也就是说，当前 CoreX 的每次 PCG solve 平均仍需要约 `36` 次迭代，而 NVIDIA 原生路径只需要约 `14.4` 次。

### 已经确认过的事实

- `12x12` block-inverse 是有效的，已经默认化。
- `4-block 3x3 structured preconditioner` 只降低约几个百分点，不能替代完整 `12x12` coupling。
- device double mixed precision 在 CoreX 上不可靠，`LDLT_DOUBLE` / `PCG_REDUCE_DOUBLE` 出现 NaN/Inf/max-iter，不能作为生产路线。
- float-only scaled LDLT 和 residual replacement 稳定，但收益接近噪声，未能进一步降低 `wb150/wb400`。

### 可能原因

当前仍无法达到 NVIDIA 原生 `38k` PCG 的原因，更可能在这些地方：

- CoreX-compatible float 路径的矩阵 assembly / reduction order 与 NVIDIA 原生路径不同，导致线性系统条件更差。
- PE-heavy contact Hessian、SPD projection 或重复/近重复接触对让部分矩阵行更难解。
- `Float` 标量轨迹、`rz/pAp/norm_r` 的舍入路径与 NVIDIA 原生高精度路径不同。
- ABD block inverse 已对齐主要结构，但剩余 contact / global matrix 部分的 preconditioning 仍弱。

## 差距 2：同一 CoreX-compatible workload 的执行成本约 1.41x

NVIDIA 硬件跑 CoreX-compatible float 路径得到：

```text
wall = 102.01s
PCG iter sum = 88085
```

CoreX 当前默认路径得到：

```text
wall = 143.956s
PCG iter sum = 88441
```

PCG workload 几乎相同，但 wall time 比例为：

```text
143.956 / 102.01 = 1.411x
```

这部分不是算法迭代数问题，而是同一兼容路径在 CoreX 运行时上的执行效率问题。可能来源包括：

- CoreX 编译器/运行时对小 kernel、高频 kernel launch、默认流同步的开销更高。
- PCG hot loop 中 SpMV、dotnorm、preconditioner apply 的单次成本更高。
- CoreX 兼容分支中保留了一些显式同步来避免错误或非确定行为，这些同步在 NVIDIA 上成本较小或不需要。
- Matrix converter、DyTopo assembly、contact filter 中的 CoreX-safe 实现通常比 NVIDIA 原生 CUB/muda 路径更保守。

这部分上限大约是 `1.4x`，不应该被误判为 `3x` 的全部差距。

## 差距 3：非 PCG 热点仍有贡献，但不是第一主因

经过前几轮优化，很多原本显著的 CoreX-only 热点已经被清掉：

- MatrixConverter linear reduce 默认化。
- AllPE 额外接触通道默认关闭。
- PCG fused `rz/norm` 默认化。
- `filter_active` 等价去同步。
- ABD DyTopo 并行 assembly。
- triangle AABB build 同步保守移除。
- PCG pinned scalar readback。

这些把 `wb400` 从约 `526s` 一路压到当前约 `144s`。

但当前与 NVIDIA 原生相比，非 PCG 部分仍可能贡献一部分差距：

- Contact filter / CCD / TOI 阶段在高 PE/EE 场景中仍重。
- Matrix conversion 和 DyTopo compaction 已大幅改善，但 CoreX-safe 路径仍比 NVIDIA 原生实现更保守。
- SpMV 和 scalar reduction 的每迭代成本仍在 PCG 总成本中占明显比例。
- 一些 host scalar/small-array readback 是算法控制流所需，不一定值得迁到 GPU。

近期 CPU fallback GPU migration 全 opt-in 组合提供了一个反例：

```text
全 opt-in GPU migration wb400: 160.741s, PCG sum 99412
默认 block-inverse wb400:      143.956s, PCG sum 88441
```

这说明“把 CPU fallback 搬到 GPU”如果改变了矩阵紧凑性、同步边界或线搜索/接触演化，可能会增加 PCG 工作量，反而更慢。

## NVIDIA 路径到底强在哪里

从目前对照看，NVIDIA 原生路径的优势不是简单的“所有计算都在 GPU 上”，而是以下组合：

1. **更好的线性系统收敛质量**  
   原生 NVIDIA 的 PCG iter sum 是 `38066`，CoreX-compatible 是约 `88k`。这是最大单项差距。

2. **更成熟的 CUDA/CUB/muda 原生实现**  
   NVIDIA 分支能安全使用更直接的 CUB segmented reduce、muda `ParallelFor` / `DeviceReduce`、stream ordering 和矩阵转换路径。CoreX 分支为了规避兼容问题，有更多保守 fallback、atomic 路径或显式同步。

3. **高精度/数值路径更可靠**  
   原生 NVIDIA 参考结果并不等价于 CoreX float 路径。此前测试还发现 NVIDIA 原生 float 本身会不稳定，说明 `45.61s / 38066` 更像高精度或更可靠数值路径的表现，而 CoreX 目前必须保持 runtime kernel 纯 `Float`。

4. **小 host 边界不是主要矛盾**  
   NVIDIA 路径中也存在一些 scalar 或 small-array host readback，例如 line-search energy scalar、TOI small array min 等。它们不一定是瓶颈。盲目改成 GPU kernel 可能因为 kernel launch、同步或下游矩阵结构变化而变慢。

## 当前差距归因

| 类别 | 估计占比/倍率 | 证据 | 优先级 |
|---|---:|---|---|
| CoreX-compatible 算法导致 PCG 迭代更多 | 约 `2.31x` PCG iter | NVIDIA CoreX-compatible `88085` vs 原生 NVIDIA `38066` | 最高 |
| 同 workload 在 CoreX 上更慢 | 约 `1.41x` wall | CoreX 当前 `143.956s` vs NVIDIA CoreX-compatible `102.01s`，PCG iter 几乎相同 | 高 |
| 非 PCG/contact/matrix 剩余热点 | 未单独精确量化 | contact、DyTopo、matrix converter、SpMV/sync 仍在 hot path | 中 |
| 未迁移 CPU fallback | 不是主因 | 全 opt-in GPU migration `160.741s`，慢于默认 | 低 |

## 建议的下一步

### 1. 做数值等价审计，而不是继续盲目 kernel sweep

建议 dump 高 PCG frame 的关键数据，例如 `wb150` frames `140-149` 或 `wb400` 高 PE/EE 区间：

- ABD `diag_hessian`
- ABD `diag_inv`
- BCOO/BSR matrix checksum 和 row statistics
- PE/PT/EE/PP selected-set ID 与 Hessian contribution distribution
- PCG scalar trace：`rz`、`pAp`、`norm_r`、`r_tol`、iter count

然后在 CPU double 或 NVIDIA 可靠高精度环境中分析：

- 当前 `diag_inv` 是否真的接近 `H^{-1}`。
- 高 PCG solve 中哪些矩阵行贡献最大。
- PE contact Hessian 是否存在重复、极端 correction/diag ratio 或 SPD projection 异常。
- CoreX matrix assembly/reduction order 是否造成比 NVIDIA 更差的 row imbalance。

### 2. 单独优化同 workload 的 1.4x 执行成本

这部分目标不是改变 PCG iter，而是让 `88k` workload 更接近 NVIDIA CoreX-compatible 的 `102s`：

- 对 PCG hot loop 做 kernel-level profiling：SpMV、dotnorm、preconditioner apply、SpMV sync。
- 优先处理能够保持数值轨迹不变的同步清理和 kernel fusion。
- 避免再次启用已证明长跑变差的 `PCG_SKIP_SPMV_SYNC`、fused SpMV-dot、full AABB async 等粗粒度开关。

### 3. 保持默认路径保守

当前最好默认路径是：

```text
wb400 = 143.956s
PCG calls = 2457
PCG iter sum = 88441
PCG iter max = 122
failure markers = 0
```

新候选只有在同时满足以下条件时才应默认化：

- `simple90/simple300/stack120` 通过。
- `wb80/wb150` 不恶化 PCG max 或 Newton/line-search 行为。
- `wb400` wall、PCG sum、PCG max 至少不退步。
- 不依赖 CoreX device double。
- 不靠 benchmark-specific frame guard。

## 总结

当前 CoreX 与 NVIDIA 的差距已经从早期“CoreX-only fallback 和串行热路径”转移到更明确的两层问题：

1. **算法/数值层**：CoreX-compatible float 路径仍需要约 `2.32x` 的 PCG 迭代数。这是从 `102s` 到 `45.6s` 的主要差距。
2. **执行层**：同一 CoreX-compatible workload 在 CoreX 上比 NVIDIA 上慢约 `1.41x`。这是从 `144s` 到 `102s` 的主要差距。

后续最有价值的方向，是用离线数值审计定位 `88k -> 38k` PCG gap 的来源，同时用 profile 精确压缩同 workload 的 `1.4x` 执行差距。机械地把剩余 CPU fallback 全部搬到 GPU 已经被 `160.741s` 的全 opt-in 结果证明不是正确主线。
