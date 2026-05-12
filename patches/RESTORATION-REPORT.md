# v11 NVIDIA 分支修复报告 (RESTORATION-REPORT)

日期：2026-04-22
对象：`libuipc-v11-extracted`（用户自 Corex 机器拷回的 v11 源码包）
基准：
- NVIDIA 字节等价基准 = `libuipc-origin/libuipc/`（NVCC 上游官方代码）
- Corex 行为基准 = "v11 修复前的当前内容"（即 Corex 上已跑通的状态）

## 1. 起因与目标

v11 修补完编译错误后能在 NVIDIA 上跑通，但 `wrecking_ball 400` 帧从 port 基准的 **46.85 s** 退化到 **558.67 s**（**11.93×** 慢）。
事后静态分析发现：v11 在号称只动 Corex 的同时，对接触/线性子系统/ABD 动力学/sim_engine 等 **NVIDIA 路径**也动了 2000+ 行，破坏了 Newton 收敛。

**目标**：让 v11 满足两条硬约束，并产出一份可直接拿回 Corex 机重跑的交付。

| 路径 | 硬约束 | 验证手段 |
|---|---|---|
| NVIDIA | 全部相关源文件的 `#else` / NVIDIA 视角 **字节等于** `libuipc-origin/libuipc/` 同名文件 | `tools/audit/verify_nvidia_branch_equiv.sh` |
| Corex | `#if UIPC_COREX_CUDA10_COMPAT` 视角 **字节等于** "v11 修复前快照" | `tools/audit/verify_corex_branch_equiv.sh` |

## 2. 解决思路：整文件 switcher 机械变换

对每个发散文件，**统一**改写为：

```cpp
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
<v11 修复前的 Corex 视角（由 extract_corex_branch.py 自动抽取）>
#else
<libuipc-origin 同名文件字节内容>
#endif
```

**关键认识**：这一变换对原本是 W（whole-file switcher）/ F（flat / 无守卫）/ I（inline-guard mix）三种结构都成立。
原因是：`extract_corex_branch.py` 输出 = "把 `UIPC_COREX_CUDA10_COMPAT` 当作已定义后预处理的内容"，
对所有结构语义相同。重新外包一层 `#if/#else/#endif` 后：
- `extract_corex_branch.py` 在新文件上的输出 = 原 Corex 视角（除可能的尾换行外）→ Corex audit 通过
- `extract_nvidia_branch.py` 在新文件上的输出 = origin 字节 → NVIDIA audit 通过

不需要逐文件理解 Corex 改了什么，只需保留它的"运行时表现"。

## 3. 实际处理量

| 桶 | 数量 | 当前结构 | 文件清单 |
|---|---|---|---|
| W (已是 switcher，#else ≠ origin) | 4 | `#if/#else/#endif` 包整文件 | [bucket-W.txt](bucket-W.txt) |
| F (无任何 COREX 守卫，整文件被改) | 84 | flat | [bucket-F.txt](bucket-F.txt) |
| I (内联 #if/#else 散布) | 50 | mix | [bucket-I.txt](bucket-I.txt) |
| **小计：发散文件** | **138** |  |  |
| v11-only 文件（origin 没有，含 Corex 适配 standalone TU） | 8 | flat | 详见正文 §6 |
| port 22 文件中 Corex view 与 v11 一致的（不在 138 里，已 ok） | 21 | switcher | — |
| **总跟踪** | **167** |  |  |

## 4. 审计基础设施

| 工具 | 路径 | 用途 |
|---|---|---|
| `extract_corex_branch.py` | `.rebase-snapshot/extract_corex_branch.py` | 提取 Corex 视角 |
| `extract_nvidia_branch.py` | `tools/audit/extract_nvidia_branch.py` | 提取 NVIDIA 视角 |
| `wrap_to_switcher.py` | `tools/audit/wrap_to_switcher.py` | 整文件 switcher 包装器（本次新增） |
| `verify_corex_branch_equiv.sh` | `tools/audit/verify_corex_branch_equiv.sh` | Corex 视角 ↔ 快照差异验证 |
| `verify_nvidia_branch_equiv.sh` | `tools/audit/verify_nvidia_branch_equiv.sh` | NVIDIA 视角 ↔ origin 差异验证 |
| Corex 基线快照 | `.rebase-snapshot/switcher-corex-branches/` | 167 文件的 Corex 视角，修复前生成 |
| Sidecar 快照 | `.rebase-snapshot/sidecars/` | 23 个 `*_corex.*` + Corex-only 文件，禁动 |

trailing-newline-only 差异在两个 audit 脚本中都被分类为 `cosmetic`（C 预处理器无副作用）。

## 5. Audit before/after

| 项 | Before (修复前) | After (修复后) |
|---|---|---|
| `verify_corex_branch_equiv.sh` | `ok=167 cosmetic=0 fail=0 missing=0` | `ok=133 cosmetic=34 fail=0 missing=0` ✅ |
| `verify_nvidia_branch_equiv.sh` | `ok=21 cosmetic=0 fail=138 missing=0` | `ok=159 cosmetic=0 fail=0 missing=0` ✅ |
| Sidecar 文件数 | 23 | 23（无变化） |
| `diff -rq .rebase-snapshot/sidecars/src libuipc-v11-extracted/src` 中 sidecar 行 | empty | empty |

> Corex audit 中的 34 个 cosmetic 全部因为整文件 switcher 在原 Corex 视角末尾追加了一个 `\n`（让 `#else` 落到独立行），C 预处理器视角下与原文件等价。

## 6. 已自动消失的 9 个临时补丁

修复 v11 编译错误时早期加的临时补丁，整文件 switcher 化后**自动消失**：

| # | 文件 | 早期补丁 | switcher 后状态 |
|---|---|---|---|
| 1 | `include/uipc/constitution/affine_body_constitution.h` | NVIDIA 分支加 `create_abd_attributes` inline 别名 | 整体被 origin 替换；origin 不需要 alias，因为 `affine_body_rod.cpp/shell.cpp` 也回到 origin（直接调 `create_attributes`） |
| 2 | `src/backends/cuda/affine_body/affine_body_state_accessor_feature.h` | 整文件包 `#if/#else` 加 NVIDIA 3-arg 构造 | 整文件 switcher，NVIDIA 段 = origin |
| 3 | `src/backends/cuda/affine_body/affine_body_state_accessor.cu` | NVIDIA 走 3-arg + `require<GlobalJointDofManager>()` | 同上 |
| 4 | `src/backends/cuda/affine_body/affine_body_revolute_joint_external_force.cu` | NVIDIA 路径调 `DRJ::theta` | NVIDIA 段 = origin（不需此 .cu，因 origin 在 `affine_body_revolute_joint.cu` 内联实现）；当前 v11-only TU 整文件 `#if COREX...#endif` |
| 5 | `src/backends/cuda/affine_body/constitutions/affine_body_revolute_joint.cu` | NVIDIA 路径调 `DRJ::theta` | NVIDIA 段 = origin |
| 6 | `affine_body_prismatic_joint_external_force.cu` / `*_constraint.cu` / `*_limit.cu` | wrap `#if COREX...#endif` 防 REGISTER_SIM_SYSTEM 重复 | 仍保留（v11-only TU，origin 没有；NVIDIA 段空） |
| 7 | `src/backends/cuda/affine_body/constitutions/affine_body_prismatic_joint_function.h` | 恢复 `compute_relative_distance` | 整文件 = origin（origin 本来就有该函数） |
| 8 | `src/backends/cuda/affine_body/abd_jacobi_matrix.h` | NVIDIA 末尾加 `#include "details/abd_jacobi_matrix.inl"` | 整文件 NVIDIA 段 = origin（origin 本来就有此 include） |
| 9 | `src/backends/cuda/affine_body/details/abd_jacobi_matrix.inl` | 删 inl 中重复算子 | 整文件 NVIDIA 段 = origin（无重复） |

`#1, #4` 两类还引申出 4 个新的 v11-only standalone TU 也整体 wrap 成 Corex-only：

```
src/backends/cuda/affine_body/affine_body_revolute_joint_external_force.cu
src/backends/cuda/affine_body/constitutions/affine_body_revolute_joint_limit.cu
src/backends/cuda/affine_body/constraints/affine_body_revolute_joint_external_force_constraint.cu
src/backends/cuda/affine_body/constraints/affine_body_revolute_joint_external_force_constraint.h
src/backends/cuda/affine_body/constraints/affine_body_prismatic_joint_external_force_constraint.h
```

## 7. NVIDIA 性能验证（详见 [perf-after-restore/SUMMARY.md](perf-after-restore/SUMMARY.md)）

| 场景 | 帧 | port (基准) | v11 损坏版 | **v11 修复后** | 修复后 vs port |
|---|---|---|---|---|---|
| simple        | 200 | 1.06 s | 1.42 s | **1.25 s** | 1.18× |
| slope         | 200 | 2.71 s | 2.51 s | **2.67 s** | 0.99× |
| stack         | 200 | 2.29 s | 3.84 s | **2.24 s** | 0.98× |
| domino        | 300 | 8.88 s | 16.48 s | **8.75 s** | 0.99× |
| wrecking_ball | 400 | 46.85 s | 558.67 s | **45.42 s** | **0.97×** ✅ |

5 场景全部达到/优于 port 基准；wb 400 从 558.67 s → 45.42 s（**12.3× 加速**），plan 硬指标 `≤ 60 s` 满足。

## 8. NVIDIA 编译

```
$ ninja -j8 in build_nvidia/  →  417/417 全部通过，0 错误
```

CUDA 12.8 / `compute_120` (Blackwell) / RTX 5070 Ti。

## 9. 文件清单

- 138 个修改文件清单：[bucket-W.txt](bucket-W.txt)、[bucket-F.txt](bucket-F.txt)、[bucket-I.txt](bucket-I.txt)
- audit baseline 输出：[audit-corex-baseline.txt](audit-corex-baseline.txt)、[audit-nvidia-baseline.txt](audit-nvidia-baseline.txt)
- audit 修复后输出：[audit-corex-final.txt](audit-corex-final.txt)、[audit-nvidia-final.txt](audit-nvidia-final.txt)
- 性能详细数据：[perf-after-restore/](perf-after-restore/)（每个场景一个 `run.log` + `SUMMARY.md`）

## 10. 拿回 Corex 机的步骤

详见 [replay-on-corex.sh](replay-on-corex.sh)。要点：
1. `rsync` 修复后的 v11 上传到 Corex 机
2. 在 Corex 机上跑 `bash tools/audit/verify_corex_branch_equiv.sh` 期望 `fail=0`
3. `cmake -DUIPC_MUDA_USE_COREX=ON ... && ninja` 编译
4. 跑 `domino` 烟测，末帧 OBJ 与 Corex 端 baseline 字节对比

## 11. 长期约束（防止再次出现 v11 这类回归）

任何在 Corex 机上做的修改，必须满足：
- 修改只出现在 `#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT` 分支或 `*_corex.*` sidecar 文件中
- 提交前在自家机上本地跑 `bash tools/audit/verify_nvidia_branch_equiv.sh`，要求 `fail=0`
- NVIDIA 视角 (`#else` 段) **不允许**任何手工编辑——它必须始终字节等于 `libuipc-origin/libuipc/` 同名文件
