# CoreX Matrix Converter 下一轮优化执行报告

## 结论

本轮确认 `MatrixConverter` 的 CoreX 兼容路径确实存在主要算法瓶颈：默认 `3x3` segmental reduce 是 `O(N * out_count)` 扫描式实现。在 `wrecking_ball150` 默认 profile 中，`matconv_kernel.segmental_reduce_3x3_scan` 单项约 `35.70s`。

新增 `UIPC_COREX_MATCONV_LINEAR_REDUCE=1` 后，`3x3/3x1` reduce 改为输入并行 atomic 累加，`wrecking_ball400` 总耗时从默认约 `553s` 降到 `505s`。该路径有明确性能收益，但 `newton_iters` 从默认约 `1996` 上升到 `2179`，暂建议保留为 opt-in，不直接默认启用。

## 实现内容

- 在 `src/backends/cuda/algorithm/details/matrix_converter.inl` 增加 `matconv.*` 子阶段 profile，拆分 sort、RLE、scan、unique、segmental reduce 等步骤。
- 在 `src/backends/cuda/dytopo_effect_system/global_dytopo_effect_manager.cu` 增加 `matconv_kernel.*` 和 `matconv_sync.*` 统计。
- 新增 `UIPC_COREX_MATCONV_LINEAR_REDUCE=1`：将 CoreX `3x3/3x1` segmental reduce 从按输出 segment 扫全量输入，改为按输入元素并行 atomic 累加到目标 segment。
- 新增 `UIPC_COREX_MATCONV_ASYNC=1`：允许跳过多数 `corex_matconv::launch_*` 后的强制 `cudaDeviceSynchronize()`，保留 trace/debug 下同步。
- 新增 `UIPC_COREX_PCG_NORM_CHECK_INTERVAL=N` 实验：尝试降低 PCG `norm(r)` 检查频率。验证后判定不安全，不作为候选。

## 验证结果

### 默认重建基线：`wrecking_ball150`

- `pcg_iters`: sum `45995`, mean `60.52`
- `newton_iters`: sum `512`, mean `3.46`
- `matconv_kernel.segmental_reduce_3x3_scan`: sum `35702.84ms`
- `dytopo.convert_matrix`: sum `32986.25ms`
- `linear.converter_convert`: sum `17361.30ms`

### `UIPC_COREX_MATCONV_LINEAR_REDUCE=1`：`wrecking_ball150`

- `simple90`: PASS
- `simple300`: PASS
- `stack120`: PASS
- `pcg_iters`: sum `46799`, mean `61.26`
- `newton_iters`: sum `516`, mean `3.49`
- `dytopo.convert_matrix`: sum `6743.38ms`
- `linear.converter_convert`: sum `1031.37ms`
- `matconv_kernel.segmental_reduce_3x3_linear`: sum `343.99ms`

### `UIPC_COREX_MATCONV_ASYNC=1`：`wrecking_ball150`

- `simple90`: PASS
- `simple300`: PASS
- `stack120`: PASS
- `pcg_iters`: sum `56944`, mean `65.30`
- `newton_iters`: sum `525`, mean `3.57`
- `dytopo.convert_matrix`: sum `29324.59ms`
- `linear.converter_convert`: sum `771.09ms`

该路径单独看 `linear.converter_convert` 很快，但 `dytopo.convert_matrix` 和 PCG/Newton 指标不如 linear reduce，暂不作为默认候选。

### `LINEAR_REDUCE + ASYNC`：`wrecking_ball150`

- `simple90`: PASS
- `simple300`: PASS
- `stack120`: PASS
- `pcg_iters`: sum `55355`, mean `64.37`
- `newton_iters`: sum `513`, mean `3.49`
- `dytopo.convert_matrix`: sum `6539.39ms`
- `linear.converter_convert`: sum `751.30ms`

组合路径的矩阵转换时间与 linear reduce 接近，但 PCG 迭代数明显更高，暂不作为 `wb400` 候选。

### `UIPC_COREX_MATCONV_LINEAR_REDUCE=1`：`wrecking_ball400`

- wall time: `505s`
- `pcg_iters`: sum `183867`, mean `63.53`
- `newton_iters`: sum `2179`, mean `5.20`
- `dytopo.convert_matrix`: sum `32978.46ms`
- `linear.converter_convert`: sum `7987.39ms`
- `matconv_kernel.segmental_reduce_3x3_linear`: sum `1642.20ms`
- `pcg.dotnorm`: sum `75779.60ms`
- `pcg.solve_total`: sum `115668.31ms`
- `contact.filter_active`: sum `74283.23ms`
- `contact.detect`: sum `63698.24ms`

## PCG Dot/Norm 实验

尝试使用 `UIPC_COREX_PCG_NORM_CHECK_INTERVAL=4` 和 `=2` 降低 `norm(r)` 检查频率。两个配置均通过 `simple90/simple300`，但 `stack120` 在第 1 帧 PCG Iter 2 出现 `NaN` 并 abort。

结论：当前 PCG 中逐迭代 `norm(r)` 检查不只是性能成本，也承担稳定性保护作用。该方向不应默认启用；后续若继续优化 PCG，应改为真正的 fused/device-scalar PCG，而不是简单降低收敛检查频率。

## 当前瓶颈迁移

matrix converter 的最大算法问题已被 `linear reduce` 明显削弱。`wb400` 当前主要瓶颈迁移到：

- `linear.assemble_linear_system`
- `pcg.solve_total`
- `pcg.dotnorm`
- `contact.filter_active`
- `contact.detect`

下一轮建议优先分析 PCG dot/norm 的 host scalar 同步和 contact filter/detect 的 CoreX/NVIDIA 路径差异。
