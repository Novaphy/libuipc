# CoreX PCG 与 Contact 下一轮优化执行报告

## 结论

本轮在上一轮 `UIPC_COREX_MATCONV_LINEAR_REDUCE=1` 基础上继续分析 PCG 与 contact 热点。最终有效候选是：

```bash
UIPC_COREX_MATCONV_LINEAR_REDUCE=1
UIPC_COREX_CONTACT_ALLPE_MODE=off
```

该组合在无 phase profile 的 `wrecking_ball400` 中耗时 `298s`，相比默认约 `553s` 和单独 `MATCONV_LINEAR_REDUCE=1` 的 `433s` 进一步下降。`simple90/simple300/stack120` 均通过。

## PCG Device-Scalar/Fused-Dot 评估

实现/验证内容：

- `corex_demo` 新增 `UIPC_COREX_LINEAR_SOLVER` 和 `UIPC_COREX_LINEAR_CHECK_INTERVAL`，用于 opt-in 切换 `linear_system/solver`。
- 使用 `UIPC_COREX_LINEAR_SOLVER=fused_pcg` 对已有 `LinearFusedPCG` 做 solver-level 对照。

验证结果：

- `simple90`: PASS
- `simple300`: PASS
- `stack120`: PASS
- `wrecking_ball150`:
  - `pcg_iters`: sum `55106`, mean `64.38`
  - `newton_iters`: sum `509`, mean `3.46`
  - `pcg.solve_total`: sum `25914.32ms`
  - `pcg.dotnorm`: sum `17010.17ms`

结论：`fused_pcg` correctness gate 通过，但 `wb150` 中 `pcg.solve_total/dotnorm` 比当前 `linear_pcg + MATCONV_LINEAR_REDUCE` 更差，因此不作为本轮候选。简单降低 norm 检查频率的 `UIPC_COREX_PCG_NORM_CHECK_INTERVAL=2/4` 此前已在 `stack120` 第 1 帧出现 NaN，也不安全。

## Contact Detect/Filter 子阶段分析

新增 `contact_detect_detail.*` 与 `contact_filter_detail.*` profile，拆分 `StacklessBVHSimplexTrajectoryFilter` 中的 AABB build、BVH build、各候选 query、各 filter kernel 和 compact/select。

`wrecking_ball50` 诊断显示：

- `contact_detect_detail.query_alle_alle`: sum `861.51ms`
- `contact_detect_detail.query_allp_alle`: sum `777.55ms`
- `contact_detect_detail.bvh_build_edge_tri`: sum `561.97ms`
- `contact_filter_detail.filter_ee`: sum `382.54ms`
- `contact_filter_detail.filter_pt`: sum `381.34ms`
- `contact_filter_detail.filter_allpe`: sum `359.29ms`
- `contact_filter_detail.select_valid_all`: sum `79.14ms`

中间候选量显示 AllPE 通道不是数量最大项，PT/EE 通常更大；但 AllPE 在每次 detect/filter 中有稳定额外 query/filter 成本，而且 NVIDIA 风格路径中没有这一条独立 AllP-AllE 通道。

## AllPE 模式实验

新增 `UIPC_COREX_CONTACT_ALLPE_MODE`：

- `full`: 当前默认行为。
- `fallback_only`: 保留 AllPE 检测，但在其他 PE 来源有效时压制 AllPE 输出。
- `off`: 跳过 AllP-AllE 独立候选通道。

### `AllPE_MODE=off + MATCONV_LINEAR_REDUCE=1`

Correctness：

- `simple90`: PASS
- `simple300`: PASS
- `stack120`: PASS

`wrecking_ball150`：

- `pcg_iters`: sum `47055`, mean `62.16`
- `newton_iters`: sum `509`, mean `3.44`
- `simplex_candidate_totals`: sum `47339780`
- `contact.detect`: sum `7193.94ms`
- `contact.filter_active`: sum `5940.91ms`

`wrecking_ball400` 无 profile：

- wall time: `298s`
- `pcg_iters`: sum `158790`, mean `62.54`
- `newton_iters`: sum `1942`, mean `4.89`
- `unique_triplets`: sum `106471108`
- `simplex_candidate_totals`: sum `122953358`

### `AllPE_MODE=fallback_only + MATCONV_LINEAR_REDUCE=1`

Correctness：

- `simple90`: PASS
- `simple300`: PASS
- `stack120`: PASS

`wrecking_ball150`：

- `pcg_iters`: sum `47453`, mean `63.27`
- `newton_iters`: sum `502`, mean `3.39`
- `simplex_candidate_totals`: sum `54791176`
- `contact.detect`: sum `10090.48ms`
- `contact.filter_active`: sum `9990.94ms`

结论：`fallback_only` 仍保留 AllPE 检测和 filter 成本，性能不如 `off`。

## Matrix Converter 复验

`UIPC_COREX_MATCONV_LINEAR_REDUCE=1` 在无 phase profile 的 `wrecking_ball400` 中：

- wall time: `433s`
- final frame 399 Newton converged at iteration 6

这说明上一轮带 phase profile 的 `505s` 被 instrumentation 放大。该候选本身有稳定收益。

## 默认启用建议

建议下一步考虑将以下两个开关作为 CoreX 默认候选，但在合入前再补一次不同场景回归：

```bash
UIPC_COREX_MATCONV_LINEAR_REDUCE=1
UIPC_COREX_CONTACT_ALLPE_MODE=off
```

理由：

- `wb400` 从默认约 `553s` 降到 `298s`。
- `simple90/simple300/stack120` 通过。
- `wb400` 的 `pcg_iters/newton_iters` 未恶化，反而低于单独 matconv linear reduce。

风险：

- `AllPE_MODE=off` 会移除 CoreX 专门增加的 AllP-AllE 独立候选通道。虽然 NVIDIA 风格路径本来也没有该通道，且当前回归通过，但仍需在更多强接触场景中验证是否存在漏检。

## 下一步

- 用 `domino`、`slope` 或更高帧数 `wrecking_ball` 回归 `AllPE_MODE=off`。
- 若接触回归稳定，再把 `MATCONV_LINEAR_REDUCE` 和 `CONTACT_ALLPE_MODE=off` 从 env opt-in 改为 CoreX 默认路径，并保留 env 回退。
- PCG 优化暂不推进到默认；后续如继续优化，应重新设计真正保留 `norm(r)` 语义的 device-side convergence check，而不是简单跳过 host 检查。
