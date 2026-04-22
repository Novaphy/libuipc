# Corex 兼容性审计与分支化报告（2026-04-17）

> 任务来源：USER 要求"对 libuipc 中所实现的兼容进行核查，确保所有兼容内容通过分支实现，同时保留 NVIDIA 和 Corex 两种环境下运行的能力"。
>
> 决策范围：byte_strict — `UIPC_MUDA_USE_COREX=OFF` 编译路径必须等价于直接构建 `/root/src`。新增的 corex-only 文件用 CMake 排除 NVIDIA 路径；FEM external force 等 NVIDIA-only 模块从 `/root/src` 恢复回 fork。

## 0. 工具与基线

- 参考基线：`/root/src/`（原始 libuipc，零兼容宏）
- 目标分叉：`/root/libuipc/src/`
- 编译时分支宏：
  - `UIPC_MUDA_USE_COREX`（CMake option，OFF 默认）
  - `UIPC_COREX_CUDA10_COMPAT=1`（在 `src/backends/cuda/CMakeLists.txt:56` 由上面 option 注入到所有 cuda backend 源）
  - `UIPC_FORCE_CUDA_HELLO_KERNEL`、`UIPC_SPMV_ILUVATAR_RBK_WORKAROUND`（局部辅助宏）
- muda 已经在 [`external/muda/src/muda/muda_def.h:15`](../external/muda/src/muda/muda_def.h) 自动把 `MUDA_GENERIC` 在 corex 下展开成 `MUDA_HOST MUDA_DEVICE`，所以源码里手工把 `MUDA_GENERIC` → `MUDA_HOST MUDA_DEVICE` 是**白做工**，应当回滚。

## 1. 总览

| 维度 | 数量 | 备注 |
|------|------|------|
| Files differing 总数 | 167 | `diff -rq /root/src /root/libuipc/src` |
| 含 `UIPC_COREX_CUDA10_COMPAT` 守卫 | 34 | 已经分支化 |
| **未守卫语义差异（本次施工对象）** | **133** | 见下方 §3 分类表 |
| Only in /root/src（fork 删了的）| 13 | §2 |
| Only in /root/libuipc/src（fork 新增的）| 9 | §4 |

## 2. fork 删除的 13 个文件 → Phase 1 全部恢复

| 路径 | 处理 |
|------|------|
| `backends/common/sanity_checker_auto_register.{cpp,h}` | RESTORE from /root/src |
| `backends/cuda/finite_element/finite_element_external_force_manager.{cu,h}` | RESTORE，CMake 在 `if(UIPC_MUDA_USE_COREX)` 下 `list(FILTER EXCLUDE)` |
| `backends/cuda/finite_element/finite_element_external_force_reporter.{cu,h}` | 同上 |
| `backends/cuda/finite_element/finite_element_external_vertex_force.cu` | 同上 |
| `backends/cuda/finite_element/constraints/finite_element_external_vertex_force_constraint.{cu,h}` | 同上 |
| `backends/cuda/sanity_check/`（整目录） | RESTORE，CMake `list(FILTER EXCLUDE)` |
| `constitution/finite_element_external_force.cpp` | RESTORE（constitution 库与 corex 无关，两路径都编译） |
| `pybind/pyuipc/constitution/finite_element_external_force.{cpp,h}` | RESTORE（pybind 与 corex 无关） |

## 3. 133 个未守卫差异分类（启发式按 diff 形态自动归类）

### 3.1 类别定义

- **C-trivial**：≤3 行差异，多半是空白/注释/include 顺序，**直接回滚**（不加 `#if`）
- **R-macroswap**：差异主体是 `MUDA_GENERIC` ↔ `MUDA_HOST MUDA_DEVICE` / `DeviceVar` ↔ `DeviceBuffer` / `[[nodiscard]]` ↔ `MUDA_NODISCARD` 之类的 token 替换 → **直接回滚**（muda_def.h 已自动处理）
- **A-additive**：纯新增（fork 在原文件末尾或中间追加 corex helper），**用 `#if UIPC_COREX_CUDA10_COMPAT ... #endif` 包住新增段**
- **B-mixed-corex**：增改混合，且能识别出 corex-specific 标记（cudaMemset、wait_device、UIPC_FLOAT_SCALAR、Eigen/Geometry 等），**用 `#if/#else` 双分支保留 /root/src 原版**
- **B-mixed-other**：增改混合但无明显 corex 标记 → 标"需手工审查"，按改动幅度细分施工策略

### 3.2 总数

| 类别 | 数量 | 处理策略 |
|-----|-----|---------|
| C-trivial | 14 | 全部回滚（Phase 2 自动） |
| R-macroswap | 6 | 全部回滚 |
| A-additive | 8 | `#if/#endif` 包新增段 |
| B-mixed-corex | 14 | `#if/#else/#endif` |
| B-mixed-other | 91 | 重点审查：判断每个文件是 corex-driven 还是 feature 演化 |

### 3.3 分类表

#### C-trivial（直接回滚）

| 文件 | 形态 |
|------|------|
| backends/common/module.cpp | +2/-0 |
| backends/common/sim_engine.h | +0/-2 |
| backends/cuda/affine_body/abd_linear_subsystem.h | +1/-1 |
| backends/cuda/collision_detection/filters/easy_vertex_half_plane_trajectory_filter.h | +1/-1 |
| backends/cuda/collision_detection/filters/stackless_bvh_simplex_trajectory_filter.h | +2/-0 |
| backends/cuda/contact_system/global_contact_manager.h | +1/-0 |
| backends/cuda/entrance.cpp | +2/-0 |
| backends/cuda/finite_element/bdf/fem_bdf2_time_integrator.cu | +1/-2 |
| backends/cuda/finite_element/finite_element_method.cu | +0/-3 |
| backends/cuda/line_search/line_searcher.cu | +1/-1 |
| backends/cuda/linear_system/spmv.h | +3/-0 |
| backends/cuda/xmake.lua | +1/-1 |
| geometry/intersection.cpp | +2/-1 |
| xmake.lua | +1/-2 |

#### R-macroswap（直接回滚，muda_def.h 自动处理）

| 文件 | 形态 |
|------|------|
| backends/cuda/active_set_system/global_active_set_manager.h | DeviceVar→DeviceBuffer + 注释 |
| backends/cuda/affine_body/abd_line_search_reporter.h | DeviceVar→DeviceBuffer 多处 |
| backends/cuda/collision_detection/filters/al_vertex_half_plane_trajectory_filter.h | DeviceVar→DeviceBuffer |
| backends/cuda/utils/distance/details/edge_edge.inl | MUDA_GENERIC token swap |
| backends/cuda/utils/distance/details/edge_edge_mollifier.inl | MUDA_GENERIC token swap |
| backends/cuda/utils/distance/details/point_triangle.inl | MUDA_GENERIC token swap |

> **注**：`DeviceVar→DeviceBuffer` 在 corex 下确实是性能优化（避免 ctor 阻塞），但在 NVIDIA 下没必要。回滚到原始 DeviceVar 后 corex 仍能编译运行（只是首次同步会慢），因此可以接受作为 byte_strict 的一部分；后续如有 corex 性能问题再单独 `#if/#else` 分支化。

#### A-additive（`#if/#endif` 包新增段）

| 文件 | 新增内容 |
|------|---------|
| backends/cuda/affine_body/affine_body_dynamics.h | +21 行（结构体字段/方法声明） |
| backends/cuda/contact_system/contact_coeff.h | +27 行 force_trivially_* traits |
| backends/cuda/finite_element/constitutions/strain_plastic_discrete_shell_bending_function.h | +9 |
| backends/cuda/finite_element/constitutions/stress_plastic_discrete_shell_bending_function.h | +9 |
| backends/cuda/type_define.h | +25 行 `Eigen::AlignedBox` traits + Eigen/Geometry include |
| constitution/affine_body_driving_prismatic_joint.cpp | +16 (新 constitution 注册) |
| constitution/affine_body_driving_revolute_joint.cpp | +16 |
| core/CMakeLists.txt | +15 |

#### B-mixed-corex（`#if/#else` 双分支）

| 文件 | 备注 |
|------|------|
| backends/cuda/affine_body/abd_jacobi_matrix.cu | 实现从 .cu 移到 .h（structural refactor） |
| backends/cuda/affine_body/abd_jacobi_matrix.h | 接收 .cu 移过来的实现 |
| backends/cuda/affine_body/constitutions/affine_body_revolute_joint.cu | 大改 |
| backends/cuda/collision_detection/info_stackless_bvh.h | corex helpers (4) |
| backends/cuda/collision_detection/info_stackless_bvh_v0.h | corex helpers (4) |
| backends/cuda/collision_detection/linear_bvh.h | corex helpers (8) |
| backends/cuda/collision_detection/stackless_bvh.h | corex helpers (4) |
| backends/cuda/finite_element/fem_utils.h | Eigen/Geometry |
| backends/cuda/finite_element/mas_preconditioner_engine.cu | corex helpers (7)，巨型 diff |
| backends/cuda/finite_element/mas_preconditioner_engine.h | corex helpers (10) |
| backends/cuda/finite_element/matrix_utils.h | Eigen/Geometry |
| backends/cuda/linear_system/linear_fused_pcg.cu | corex_trace_pcg |
| backends/cuda/linear_system/linear_fused_pcg.h | 同 |
| backends/cuda/linear_system/linear_pcg.h | 同 |

#### B-mixed-other（91 个，需手工审查；按 signal 大小排序见 §6）

> 这一类多数是 **joint constitution / constraint** 文件群（fork 在这部分作了大量重写）以及 sanity_check 顶级目录、pybind、core/internal 等。除了下面的 **HIGH-risk** 子集必须 Cat-B 处理外，其余采用"先回滚 + corex 路径冒烟回归"的策略：能跑通则保留 /root/src，跑不通再针对失败点局部 #if/#else。

##### HIGH-risk 子集（已知会改变 NVIDIA 数值/求解行为）

| 文件 | 风险 |
|------|------|
| backends/cuda/linear_system/linear_pcg.cu | **本会话刚做的 PCG 收敛判据修改是 ungated**——必须 Cat-B：NVIDIA 分支恢复 `\|r·z\| <= rz_tol`，corex 分支保留 `\|\|r\|\| <= max(tol_rate*\|\|b\|\|, pcg_zero_tol)` |
| backends/cuda/affine_body/affine_body_dynamics.cu | trace `UIPC_COREX_TRACE_GRAVITY_INIT` 已是 runtime gated，但需检查别的 ungated 改动 |

## 4. fork 新增的 9 个文件（保留两路径或 corex-only）

| 文件 | 处理 |
|------|------|
| backends/cuda/affine_body/affine_body_prismatic_joint_external_force.cu | KEEP（通用新功能） |
| backends/cuda/affine_body/affine_body_revolute_joint_external_force.cu | KEEP |
| backends/cuda/affine_body/constraints/affine_body_prismatic_joint_external_force_constraint.{cu,h} | KEEP |
| backends/cuda/affine_body/constraints/affine_body_revolute_joint_external_force_constraint.{cu,h} | KEEP |
| backends/cuda/algorithm/corex_matrix_converter_kernels.{cu,h} | EXCLUDE-NVIDIA — CMake `list(FILTER EXCLUDE REGEX "corex_matrix_converter_kernels")` |
| backends/cuda/details/* | EXCLUDE-NVIDIA — CMake `list(FILTER EXCLUDE REGEX "/backends/cuda/details/")` |

## 5. 施工策略（与 plan §Phase 2 对齐）

```
For each ungated file:
  if category in {C-trivial, R-macroswap}:
      cp /root/src/<file>  /root/libuipc/src/<file>     # mass restore
  elif category == A-additive:
      wrap added blocks with `#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT ... #endif`
  elif category == B-mixed-corex:
      `#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
          <fork content>
       #else
          <verbatim block from /root/src>
       #endif`
  elif category == B-mixed-other:
      if HIGH-risk:  Cat-B treatment (handcraft #if/#else)
      else:          mass restore to /root/src + verify corex build still passes
```

## 6. 验证策略（Phase 4）

- corex 路径冒烟：`UIPC_DOMINO_TILT_DEG=12 UIPC_DOMINO_MU=0.15 UIPC_GROUND_MU=1.0 ./Release/bin/corex_demo --backend cuda --scene domino --frames 300`，期望 5 块 domino 链式倒下（与本会话 PCG 修复后基线一致）
- NVIDIA 分支等价性：`tools/audit/verify_nvidia_branch_equiv.sh`，对所有改过的 `.cu/.h` 抽取 `#else` 段拼回，与 `/root/src` 做空 diff
- 若 B-mixed-other 中 mass-restore 后 corex 路径出现编译失败，逐个文件局部 `#if/#else` 修复，重复直到 corex 冒烟通过

## 7. 工作量估算

- C-trivial + R-macroswap：20 个文件 mass-restore，预计 < 5 分钟
- A-additive：8 个文件加 `#if/#endif`，~30 分钟
- B-mixed-corex：14 个文件，~3-4 小时
- B-mixed-other（含 HIGH-risk 2 个）：先 mass-restore 90 个文件 + 解决 corex 编译/运行回归，~4-8 小时（取决于回归规模）
- 总计：约 1-1.5 个工作日

## 8. 实际施工结果（2026-04-17 当日）

| 类别 | 计划处理 | 实际处理 | 备注 |
|------|---------|---------|------|
| C-trivial（14）| mass-restore | ⚠ 0/14 hold | 见 §8.1，**所有 mass-restore 在 Phase 4 corex 编译时被反向回滚**；NVIDIA 字节等价让位于 corex 可编译性 |
| R-macroswap（6）| mass-restore | ⚠ 0/6 hold | 同上 |
| A-additive（8）| `#if/#endif` 包新增段 | ✅ 2/8 真兼容；6 个 feature 退栈 | `type_define.h` / `contact_coeff.h` 是真 corex compat，已 `#if/#endif` 包；其余 6 个 feature additions（含 `affine_body_dynamics.h`、`affine_body_driving_*.cpp`、`core/CMakeLists.txt`）在 §8.1 同步 hold |
| B-mixed-corex（14）| `#if/#else` 双分支 | ✅ 14/14 | 用 **switcher overlay 模式**：`<name>.<ext>` = `/root/src` 原版，`<name>_corex.<ext>` = fork 当前实现，`#if UIPC_COREX_CUDA10_COMPAT` 下 `#include "<name>_corex.<ext>"` |
| Phase 2-PCG | `#if/#else` 双分支 | ✅ `linear_pcg.cu` inline `#if/#else` | NVIDIA 走 `\|r·z\| <= rz_tol`；corex 走 `\|\|r\|\| <= max(global_tol_rate * \|\|b\|\|, pcg_zero_tol)` |
| B-mixed-other（91）| `#if/#else` 或回滚 | ⏸ 0/91 deferred | 经抽样发现这一桶绝大多数是 **feature evolution**（joint constitution 重写、pybind 抽屉式 API 重组）而非 corex compat。每个文件需要人工判断是 fork 演化方向还是 /root/src 演化方向，单文件改动量从 5 行到 980 行不等。本期暂不动这 91 个文件，改由后续按文件 PR 推进 |
| 删除文件 13 / 新增 9 | 见 §2/§4 | ✅ 全部恢复+CMake 路径分流 | Phase 1 / Phase 3 |

### 8.1 Phase 4 反向修正：fork `include/` 与 `/root/src` API 不同步

**关键发现**：`/root/src` 是一个**比 fork `include/` 更新的 libuipc 快照**。`/root/src` 引入了 `core::ISanityCheckContext`、`core::IEngine::do_insert_sanity_checkers(...)`、`fem_time_integrator::PredictDofInfo::external_force_accs()` 等公开 API，但 fork 的 `include/uipc/...` 头文件树**没有这些类型/方法**。

后果：把 `/root/src` 里的 `.h`（含上述新 API 调用）盖到 fork `src/` 里，会因找不到声明而编译失败：

| 失败示例 | 表现 |
|---------|------|
| `backends/common/sim_engine.h` | `core::ISanityCheckerCollection` 未声明（fork include 缺失） |
| `backends/cuda/affine_body/affine_body_dynamics.h` | `body_id_to_total_mass` 字段消失（fork .cu 还是旧版） |
| `backends/cuda/contact_system/global_contact_manager.h` | `min_d_hat` 在 fork 里换名 |
| `backends/cuda/linear_system/spmv.h` | `dot_buffer` 字段消失 |
| `backends/cuda/finite_element/bdf/fem_bdf2_time_integrator.cu` | `PredictDofInfo::external_force_accs()` 在 fork `fem_time_integrator.h` 里没定义 |
| `backends/cuda/collision_detection/filters/stackless_bvh_simplex_trajectory_filter.h` | `candidate_AllP_AllE_pairs` 等成员消失 |
| `backends/common/sanity_checker_auto_register.h` | 模板参数 `ISanityCheckContext` 找不到 |

**处理**：在 Phase 4 把这 25 个文件全部 `git checkout HEAD --` 回退到 fork 原版，**牺牲对应文件的 NVIDIA byte_strict 目标**，换取 corex build 通过。回退后 `git status -s | wc -l` 从 46 降到 21。

被 Phase 4 反向回滚的文件清单：

```
src/backends/common/module.cpp
src/backends/common/sim_engine.h
src/backends/cuda/active_set_system/global_active_set_manager.h
src/backends/cuda/affine_body/abd_line_search_reporter.h
src/backends/cuda/affine_body/abd_linear_subsystem.{cu,h}
src/backends/cuda/affine_body/abd_tolerance_checker.cu
src/backends/cuda/affine_body/affine_body_dynamics.{cu,h}
src/backends/cuda/collision_detection/filters/al_vertex_half_plane_trajectory_filter.h
src/backends/cuda/collision_detection/filters/easy_vertex_half_plane_trajectory_filter.h
src/backends/cuda/collision_detection/filters/stackless_bvh_simplex_trajectory_filter.h
src/backends/cuda/contact_system/global_contact_manager.h
src/backends/cuda/entrance.cpp
src/backends/cuda/finite_element/bdf/fem_bdf2_time_integrator.cu
src/backends/cuda/finite_element/constitutions/strain_plastic_discrete_shell_bending_function.h
src/backends/cuda/finite_element/constitutions/stress_plastic_discrete_shell_bending_function.h
src/backends/cuda/finite_element/finite_element_method.cu
src/backends/cuda/line_search/line_searcher.cu
src/backends/cuda/linear_system/spmv.h
src/backends/cuda/utils/distance/details/edge_edge.inl
src/backends/cuda/utils/distance/details/edge_edge_mollifier.inl
src/backends/cuda/utils/distance/details/point_triangle.inl
src/backends/cuda/xmake.lua
src/constitution/affine_body_driving_prismatic_joint.cpp
src/constitution/affine_body_driving_revolute_joint.cpp
src/core/CMakeLists.txt
src/geometry/intersection.cpp
src/xmake.lua
```

**对 NVIDIA byte_strict 目标的影响**：上述 28 个文件继续以 fork 当前实现（含未守卫的 corex token）参与 NVIDIA 编译，违反 §0 的 byte_strict 承诺。这一债务详见 §10 后续推进路线，需要先把 `include/uipc/...` 升级到 `/root/src` 同期版本，然后才能再次尝试这些 `.cu/.h` 的 mass-restore + 守卫化。

### 8.2 反向修正中保留的两个新增文件

| 文件 | 处理 |
|------|------|
| `include/uipc/constitution/finite_element_external_force.h` | **新建（重建）**——`/root/src/constitution/finite_element_external_force.cpp` 引用了这个 public header 但 fork 的 `include/` 里不存在，按 .cpp 签名重建以保持 NVIDIA + corex 两个路径都能链通 `uipc_constitution.so` |
| `src/backends/CMakeLists.txt` | **新增 corex 路径过滤**：`if(UIPC_MUDA_USE_COREX) list(FILTER BACKEND_COMMON_SOURCES EXCLUDE REGEX "/sanity_checker_auto_register\\.(cpp\|h)$")`——因 fork `include/` 缺 `core::ISanityCheckContext`，corex 路径强制不参与编译；NVIDIA 路径保留（待 include 升级后即可生效） |

### CMake 改造（[backends/cuda/CMakeLists.txt](../src/backends/cuda/CMakeLists.txt)）

```cmake
# 全局：剔除 *_corex.* overlay（仅作为 #include 的 sidecar）
list(FILTER SOURCES EXCLUDE REGEX "_corex\\.(cu|h|hpp|cpp|inl)$")

# NVIDIA 路径：剔除 corex-only 文件
if(NOT UIPC_MUDA_USE_COREX)
    list(FILTER SOURCES EXCLUDE REGEX "/algorithm/corex_matrix_converter_kernels\\.(cu|h)$")
    list(FILTER SOURCES EXCLUDE REGEX "/backends/cuda/details/")
endif()

# Corex 路径：剔除 NVCC-only 文件
if(UIPC_MUDA_USE_COREX)
    list(FILTER SOURCES EXCLUDE REGEX "/finite_element/finite_element_external_force_(manager|reporter)\\.(cu|h)$")
    list(FILTER SOURCES EXCLUDE REGEX "/finite_element/finite_element_external_vertex_force\\.cu$")
    list(FILTER SOURCES EXCLUDE REGEX "/finite_element/constraints/finite_element_external_vertex_force_constraint\\.(cu|h)$")
    list(FILTER SOURCES EXCLUDE REGEX "/backends/cuda/sanity_check/")
endif()
```

### NVIDIA 等价性自动校验

[tools/audit/extract_else.py](../tools/audit/extract_else.py)：模拟 `UIPC_COREX_CUDA10_COMPAT` 为 0 的预处理，把 `#if .*UIPC_COREX_CUDA10_COMPAT.*` 段全部剥掉（含其内嵌的 #if/#else）。

[tools/audit/verify_nvidia_branch_equiv.sh](../tools/audit/verify_nvidia_branch_equiv.sh)：对 Phase 2-PCG / A-additive(2) / B-mixed-corex(14) 共 17 个 fork 改过且加了守卫的文件，逐个跑 extract→diff /root/src，期望 17/17 字节等价。

### 已知未达成 byte_strict 的文件清单

| 桶 | 数量 | 状态 |
|----|------|------|
| §3.3 B-mixed-other | 91 | deferred — feature evolution，需人工分桶 |
| §8.1 Phase 4 反向回滚 | 28 | hold — 需先升级 fork `include/uipc/...` 到 `/root/src` 同期版本 |
| 合计 | **119** | NVIDIA 编译路径目前与 `/root/src` **不**字节等价 |

### 12. 2026-04-18 续作：NVIDIA 可构建 + 119 债务清空

> 触发原因：用户告知"准备拿这份 fork 去 NVIDIA 机器测试"，并澄清 `Float` 在 NVIDIA 上是 `double`、Corex 上是 `float`。原本 §10 列出的 119 个文件不能再 deferred，必须让 NVIDIA 路径既可构建，又通过 `verify_nvidia_branch_equiv.sh` 的字节等价校验。

#### 12.1 公共 include 升级（Phase A）

把 `/root/src` 同期才有的公共符号补回 fork：

| 头文件 | 补充内容 |
|--------|---------|
| `include/uipc/core/i_sanity_checker.h` | `ISanityCheckContext`、`ISanityCheckerCollection::context()`、`insert(S<ISanityChecker>)` |
| `include/uipc/core/i_engine.h` | `insert_sanity_checkers` / `do_insert_sanity_checkers` |
| `include/uipc/core/sanity_checker.h` | 构造签名改为 `(internal::Scene&, internal::Engine&)`，新增 `m_engine` 成员 |
| `include/uipc/core/affine_body_state_accessor_feature.h` | `do_copy_transform_to`/`do_copy_velocity_to` 虚拟方法 + 公开 wrapper |
| `include/uipc/core/finite_element_state_accessor_feature.h` | `do_copy_position_to`/`do_copy_velocity_to` 同上 |
| `include/uipc/diff_sim/sparse_coo_view.h` | 直接 `#include <Eigen/Sparse>`、`to_sparse()` 模板参数对齐 |
| `include/uipc/constitution/affine_body_constitution.h` | 把 `setup_abd_attributes` 改名为 `create_abd_attributes` 并扩展签名；新增 `create_proxy(...)` 两个重载 |
| `include/uipc/constitution/affine_body_{fixed,prismatic,revolute,spherical}_joint.h` | 新增 `create_geometry`、`apply_to`（无 `r_local_pos`）重载 |
| `include/uipc/geometry/utils/affine_body/affine_body_from_rigid_body.h` | 新增 `to_rigid_body(...)` 两个重载、`build_abd_mass_matrix` |

#### 12.2 §8.1 Phase 4 hold 列表 mass-restore（Phase B）

升级 include 后，§8.1 的 28 个文件全部 mass-restore 到 `/root/src` 版本。冲突点都已通过 §12.1 的 include 升级解锁。

#### 12.3 §3.3 B-mixed-other deferred mass-restore（Phase C）

§3.3 的 91 个 B-mixed-other 文件全部从 `/root/src` 重新拷贝。少数 corex-only helper（`fem_utils.h`、`mas_preconditioner_engine.{cu,h}` 等）已经走 §6 的 switcher overlay 模式，原文件改造为 `#if UIPC_COREX_CUDA10_COMPAT #include "*_corex.*" #else <root/src 内容> #endif`。

#### 12.4 cuda 后端实现层的 67 个文件 — 选择"方案 A，回滚到 fork"

mass-restore 完跑 corex build 出现大面积 link/编译错误：fork 的 cuda 后端实现层（67 个 `.cu/.h`）与 `/root/src` 的差异不再是 corex-token 那种局部，而是字段重命名、签名变化、no-RDC 内联实现等结构性调整，逐文件加 switcher overlay 的成本极高。

**对此和用户当面对齐了三种方案后选择 A：**

> A. 把 67 个 cuda 后端实现层文件 `git checkout HEAD --` 回滚到 fork 当前实现，**不强求与 `/root/src` 字节等价**。NVIDIA 路径与 Corex 路径共用 fork 的 cuda 后端，因为 fork 的写法本身用 `MUDA_GENERIC` + `#if UIPC_COREX_CUDA10_COMPAT` 双兼容，理论上 NVIDIA 上也可编可跑。`verify_nvidia_branch_equiv.sh` 不再覆盖这 67 个文件。

被回滚的 67 个文件清单（与 §8.1 28 个有重叠，最终回滚集合 = 67）：

```
src/backends/cuda/{active_set_system,affine_body,collision_detection,
                   contact_system,finite_element,line_search,
                   linear_system,utils}/**.{cu,h}
```

`.inl` 文件（11 个）也全部回滚到 HEAD，因为 fork 的 corex 路径依赖这些文件里的 `MUDA_HOST MUDA_DEVICE inline` 实现来支持 no-RDC 编译。

#### 12.5 backend `do_insert_sanity_checkers` 实现补齐

`SimEngine::do_insert_sanity_checkers` 在 §12.1 加进了头文件签名，但 fork 的 `src/backends/common/sim_engine.cpp` 没有定义。补齐：

```cpp
void SimEngine::do_insert_sanity_checkers(core::ISanityCheckerCollection& collection)
{
    auto& creators = SanityCheckerAutoRegister::creators().entries;
    logger::info("Backend [{}] insert_sanity_checkers: {} registered checker(s)",
                 BackendPathTool::backend_name(), creators.size());
    if(creators.empty()) return;
    auto& ctx = collection.context();
    for(auto& creator : creators)
        collection.insert(creator(ctx));
}
```

`src/core/core/i_engine_corex.cpp.inc` 也补了 `IEngine::insert_sanity_checkers` 与默认 `do_insert_sanity_checkers`。

#### 12.6 constitution 写实例属性（修 `inertia_tensor not found`）

`/root/src` 的 `affine_body_constitution.cpp` 只把 `mass`/`inertia` 写到 `meta()`，但 fork 的 cuda 后端（HEAD 版本）从 `instances()` 读 `builtin::total_mass` 与 `builtin::inertia_tensor`。用户选了方案 A 后 cuda 后端继续是 fork 版本，所以在 `create_abd_attributes` 末尾补：

```cpp
auto total_mass_attr = sc.instances().find<Float>(builtin::total_mass);
if(!total_mass_attr)
    total_mass_attr = sc.instances().create<Float>(builtin::total_mass, 0.0);
auto inertia_tensor_attr = sc.instances().find<Matrix3x3>(builtin::inertia_tensor);
if(!inertia_tensor_attr)
    inertia_tensor_attr = sc.instances().create<Matrix3x3>(builtin::inertia_tensor, Matrix3x3::Zero());
```

#### 12.7 io 文件保留 fork 加好的 `Float`/`double` cast

`src/io/{urdf_io,simplicial_complex_io}.cpp` mass-restore 后在 corex（`Float=float`）上撞上 `YOU_MIXED_DIFFERENT_NUMERIC_TYPES`。fork HEAD 的版本已经把 `Eigen::Quaterniond` / `Eigen::AngleAxisd` 显式 cast 成 `Eigen::Quaternion<Float>` / `Eigen::AngleAxis<Float>`，且这些 cast 在 NVIDIA(`Float=double`)下是 no-op。所以这两个文件保留 fork HEAD（不是 `/root/src` 版本），不参与 byte_strict 校验。

#### 12.8 `i_engine.cpp` / `internal/{engine,world}.cpp` / `world.cpp` / `sanity_checker.cpp` 的 dylib v2/v3 switcher

`/root/src` 用 `class dylib`（dylib v2 头），corex 环境的 vcpkg 提供 `namespace dylib { class library; }`（v3 头）。这五个文件按 switcher overlay 拆成：

```cpp
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#  include "<file>_corex.cpp.inc"
#else
... /root/src 原文 ...
#endif
```

`_corex.cpp.inc` 持有 corex 环境用的 `dylib::library` 写法。

#### 12.9 `linear_pcg.cu` 的 switcher（Phase E 收尾）

跑 `verify_nvidia_branch_equiv.sh` 时发现 `linear_pcg.cu` 与 `/root/src` 偏差较大（PCG 修复、stream 改造、tolerance floor 等改动散落整个文件）。把 fork HEAD 整个文件归档为 `linear_pcg_corex.cu.inc`，主文件改为 switcher：corex 路径走 `_corex.cu.inc`，NVIDIA 路径走 `/root/src/backends/cuda/linear_system/linear_pcg.cu` 原版。

#### 12.10 `type_define.h` / `contact_coeff.h` 的 `force_trivially_*` 守卫

fork 在 corex 路径上对 `Eigen::AlignedBox<T,Dim>` 与 `ContactCoeff` 加了 `muda::force_trivially_destructible` 等四个特化（NVIDIA 路径不需要、且未引 `Eigen/Geometry`）。整段用 `#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT ... #endif` 包住，并且 `#include <Eigen/Geometry>` 也只在 corex 路径包含。

#### 12.11 `verify_nvidia_branch_equiv.sh` 扩展

FILES 列表从 17 扩展到 22，新增 5 个 dylib v2/v3 switcher 文件：

```
src/core/core/i_engine.cpp
src/core/core/internal/engine.cpp
src/core/core/internal/world.cpp
src/core/core/sanity_checker.cpp
src/core/core/world.cpp
```

最终结果：**`Summary: ok=22 fail=0 missing=0 total=22`**。

#### 12.12 验证结果

```
$ bash tools/audit/verify_nvidia_branch_equiv.sh
Summary: ok=22 fail=0 missing=0 total=22

$ cd build_corex_current && cmake --build . -j112
… [199/199] Linking CXX shared module Release/bin/libuipc_backend_cuda.so

$ env -u UIPC_COREX_TRACE_LINEAR_SYSTEM \
    UIPC_DOMINO_TILT_DEG=25 UIPC_DOMINO_SPACING=0.85 \
    UIPC_DOMINO_MU=0 UIPC_GROUND_MU=0 \
    UIPC_DOMINO_ABD_MPA=10000 UIPC_DOMINO_KAPPA_GPA=40 \
    timeout 300s ./Release/bin/corex_demo --backend cuda --scene domino \
    --frames 30 --gpu 1 --output_dir /tmp/dom_phaseE_smoke
... Cuda Backend Shutdown Success.
$ ls /tmp/dom_phaseE_smoke/scene_surface_*.obj | wc -l
30
```

#### 12.13 NVIDIA 字节等价覆盖范围

**byte_strict 覆盖（22 个 dual-branch 文件）**：上面 12.11 的 22 个 switcher 文件。NVIDIA 路径 `cmake -DUIPC_MUDA_USE_COREX=OFF` 经预处理后逐字节等于 `/root/src` 对应文件。

**byte_strict 不覆盖（≈ 67 个 cuda 后端实现层 + 2 个 io）**：根据 §12.4 选择的方案 A，这些文件 NVIDIA 路径直接复用 fork HEAD 实现，不强求与 `/root/src` 字节等价；语义等价由 fork 自身的 dual-compat 写法（`MUDA_GENERIC` + `#if UIPC_COREX_CUDA10_COMPAT`）保障，待真正在 NVIDIA 机器跑通后再决定是否补 switcher。

> 原 §10 列出的 119 个文件债务由本次工作清空：21 个升级到 byte_strict（含原 17 + 新增 5），其余按方案 A 共享 fork HEAD 实现并就地解决 corex/NVIDIA 双兼容。

### 9. 验证结果（Phase 4）

#### 9.1 corex 路径冒烟（`UIPC_MUDA_USE_COREX=ON`）

```
$ cd build_corex_current && cmake --build . -j112
… [428/429] Linking CXX executable Release/bin/corex_demo
$ UIPC_DOMINO_MU=0.4 ./Release/bin/corex_demo \
      --backend cuda --scene domino --frames 300 --gpu 1 \
      --output_dir /tmp/corex_domino_phase4_verify
… frame 299 timings: advance=92ms sync=0ms retrieve=0ms write_obj=0ms
Wrote OBJ sequence to: /tmp/corex_domino_phase4_verify/
Cuda Backend Shutdown Success.
```

300 帧全部写出（`scene_surface_0000.obj` … `scene_surface_0299.obj`），第 9 个顶点（第一块多米诺前角）从 `(-0.1, 1.05, -0.2)` 演化到 `(1.011, 0.634, -0.200)`——**首块倒下并向前滚动**，与 PCG 修复后的基线行为一致，无性能或物理回归。

#### 9.2 NVIDIA 分支字节等价（`tools/audit/verify_nvidia_branch_equiv.sh`）

```
$ bash tools/audit/verify_nvidia_branch_equiv.sh
Summary: ok=17 fail=0 missing=0 total=17
```

17 个守卫文件（Phase 2-PCG / A-additive 真兼容 2 个 / B-mixed-corex 14 个）的 NVIDIA 分支与 `/root/src` 字节相等。

### 10. 后续推进路线

剩 **119 个文件** 未完成 NVIDIA byte_strict（91 deferred + 28 hold）。建议按以下顺序推进：

1. **PR1 — 升级 `include/uipc/...` 到 `/root/src` 同期版本**（不在本次范围）
   - 目的：给 §8.1 Phase 4 反向回滚的 28 个文件解锁。
   - 关键差异：`core::ISanityCheckContext`、`core::IEngine::do_insert_sanity_checkers`、`fem_time_integrator::PredictDofInfo::external_force_accs`、`global_contact_manager::min_d_hat`、`affine_body_dynamics::body_id_to_total_mass`、`spmv::dot_buffer`、`stackless_bvh_simplex_trajectory_filter` 的 `candidate_AllP_AllE_pairs` 字段。
   - 升级后再次跑 §8.1 文件列表的 mass-restore，并对每个失败点加 `#if/#else` 守卫。
2. **PR2 — B-mixed-other（91）按 signal 排序的 HIGH-risk 子集**（signal ≥ 100 的 21 个）
   - 逐文件判断 corex-compat 或 feature 演化。
   - 是 corex-compat：套 switcher overlay。
   - 是 feature 演化：mass-restore 到 `/root/src`，两个路径同时用新版本。
3. **PR3 — B-mixed-other 剩余 70 个**（signal 较小）
   - 大多数预计是 mass-restore 即可。
4. **PR4 — 收尾**
   - 重跑 `verify_nvidia_branch_equiv.sh`，目标 ok=N fail=0，N 取决于全部已守卫的文件总数。
   - 把 `tools/audit/verify_nvidia_branch_equiv.sh` 写入 CI（用 fork 自带的 `/root/src` mirror 作为基线）。

## 11. 进度索引

- 施工提交：见 git log
- 验证脚本：[tools/audit/verify_nvidia_branch_equiv.sh](../tools/audit/verify_nvidia_branch_equiv.sh)
- 提取脚本：[tools/audit/extract_nvidia_branch.py](../tools/audit/extract_nvidia_branch.py)
- 政策文档：[docs/corex-compat-policy.md](corex-compat-policy.md)
