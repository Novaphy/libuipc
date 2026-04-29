# CoreX v14 默认化与下一轮瓶颈执行报告

## 结论

本轮将 v13 已验证的两个高收益候选改为 CoreX 默认路径：

- MatrixConverter 默认使用 input-linear segmental reduce。
- StacklessBVHSimplexTrajectoryFilter 默认关闭独立 AllP-AllE / AllPE 候选通道。

默认配置下无需额外环境变量，`wrecking_ball400` 为 `308s`，`wrecking_ball800` 为 `595s`。这与 v13 opt-in 配置基本一致。

## 默认化实现

### MatrixConverter

默认行为改为 linear reduce。旧扫描路径通过以下环境变量回退：

```bash
UIPC_COREX_MATCONV_SCAN_REDUCE=1
```

### Contact AllPE

默认不生成独立 AllP-AllE / AllPE 通道。完整旧路径可通过以下环境变量回退：

```bash
UIPC_COREX_CONTACT_ALLPE_MODE=full
```

仍保留：

```bash
UIPC_COREX_CONTACT_ALLPE_MODE=fallback_only
```

用于诊断保留 AllPE 检测但压制冗余 AllPE 输出的折中路径。

## 默认回归结果

结果目录：

- `/root/corex-v14-default-regression-2026-04-29/`
- `/root/corex-v14-default-wb800-2026-04-29/`

五场景默认回归：

- `simple200`: `6s`, exit `0`
- `slope200`: `12s`, exit `0`
- `stack200`: `9s`, exit `0`
- `domino300`: `44s`, exit `0`
- `wrecking_ball400`: `308s`, exit `0`

`wrecking_ball800`：

- wall time: `595s`
- exit code: `0`
- `pcg_iters`: sum `323220`, mean `60.80`
- `newton_iters`: sum `4319`, mean `5.42`
- `unique_triplets`: sum `226618224`, mean `21314.73`
- `simplex_candidate_totals`: sum `157016903`, mean `11154.15`

## PCG Fused pAp 实验

新增 opt-in：

```bash
UIPC_COREX_PCG_FUSED_PAP=1
```

该路径只将 `pAp = p^T A p` 与 SpMV 融合，不改变 `norm(r)` 收敛检查语义。

验证结果：

- `simple90`: PASS
- `simple300`: PASS
- `stack120`: PASS
- `wrecking_ball150`:
  - `pcg_iters`: sum `63402`
  - `newton_iters`: sum `493`
  - `pcg.solve_total`: sum `16172.09ms`
  - `pcg.dotnorm`: sum `8534.91ms`

结论：虽然 correctness gate 通过，但 `pcg_iters` 和整体 `pcg.solve_total` 不优于默认路径，因此不默认启用。

## Contact PT/EE 热点分析

AllPE 默认关闭后，`wrecking_ball80` profile 显示剩余 contact 主要来自：

- `contact_detect_detail.query_alle_alle`: sum `1422.99ms`
- `contact_detect_detail.bvh_build_edge_tri`: sum `915.00ms`
- `contact_filter_detail.filter_pt`: sum `643.66ms`
- `contact_filter_detail.filter_ee`: sum `642.02ms`
- `contact_detect_detail.query_allp_allt`: sum `607.67ms`

候选统计：

- `PT_cands`: sum `10767139`
- `EE_cands`: sum `25505237`
- `AllPE_cands`: sum `0`
- sampled valid `PT`: sum `5493`
- sampled valid `EE`: sum `60553`
- sampled valid `PE`: sum `54425`

这说明 PT/EE 候选规模仍远大于最终有效输出。下一步如果继续优化 contact，应优先考虑将部分 active-distance 判据前移到 detect query predicate，但这属于候选裁剪，必须保持 opt-in 并做强正确性回归。

## 强接触回归

`domino600` 默认配置通过：

- wall time: `60s`
- exit code: `0`
- `pcg_iters`: sum `77316`
- `newton_iters`: sum `1255`
- `simplex_candidate_totals`: sum `273455`
- 最后一帧仍有 PE 接触：`PTs=18, EEs=23, PEs=45, PPs=0`

结合 `wb800`，当前默认 AllPE off 没有在长帧数和 PE 活跃接触中暴露崩溃或不收敛。

## 默认启用建议

v14 默认化可以保留：

- Matrix linear reduce 默认启用。
- AllPE independent channel 默认关闭。

保留的风险：

- AllPE off 仍然是候选裁剪。虽然 NVIDIA 风格路径没有该通道且当前回归通过，但如果后续遇到点-边接触漏检，应优先用 `UIPC_COREX_CONTACT_ALLPE_MODE=full` 回退确认。

## 后续优化方向

- Contact：对 `query_alle_alle`、`filter_pt/filter_ee` 做更细致的候选有效率优化。
- PCG：不要再跳过 `norm(r)`；若继续优化，应做 device-side convergence check，而不是减少检查频率。
- Assembly：在默认路径稳定后，再继续拆 `linear.assemble_linear_system` 中 ABD/contact/dytopo 的剩余占比。
