# Corex / NVIDIA 双兼容代码政策

> 适用范围：`/root/libuipc` 全仓
>
> 起草时间：2026-04-17，于 2026-04-18 修订，于 **2026-04-22 重新对齐 baseline**
>
> 关联报告：[corex-compat-audit-2026-04-17.md](corex-compat-audit-2026-04-17.md)、[rebase-2026-04-22.md](rebase-2026-04-22.md)
>
> **2026-04-22 baseline 升级**：NVIDIA 分支的"参考实现 baseline"由旧的 `/root/src` 升级为 `libuipc-origin/libuipc/` 在 2026-04-22 抓取的快照（与 origin upstream 同步）。所有 22 个 switcher 文件的 NVIDIA 分支字节等价于该快照（`tools/audit/verify_nvidia_branch_equiv.sh` 22/22 OK），所有 23 个 corex sidecar 文件与 22 个 switcher 的 corex 分支与 rebase 前快照字节等价（`tools/audit/verify_corex_branch_equiv.sh` 21 OK + 1 cosmetic）。

## 1. 目标

让同一份源码可以编译/运行于两个完全不同的 GPU 平台：

| 路径 | CMake option | 关键宏 | `Float` |
|------|--------------|-------|---------|
| **NVIDIA**（参考实现 = `/root/src`） | `-DUIPC_MUDA_USE_COREX=OFF`（默认） | `UIPC_COREX_CUDA10_COMPAT` 未定义 | `double`（`UIPC_USE_FLOAT=OFF`） |
| **天数 Corex** | `-DUIPC_MUDA_USE_COREX=ON` | `UIPC_COREX_CUDA10_COMPAT=1` | `float`（`UIPC_USE_FLOAT=ON`） |

任何 corex-only / NVIDIA-only 的源码差异**必须**通过分支化表达。

## 2. 强制守卫宏

唯一允许在源码里出现的兼容分支宏是：

```cpp
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
   // corex-only
#else
   // NVIDIA / 通用
#endif
```

`UIPC_COREX_CUDA10_COMPAT` 由 [src/backends/cuda/CMakeLists.txt](../src/backends/cuda/CMakeLists.txt) 与 [src/core/CMakeLists.txt](../src/core/CMakeLists.txt) 在 `if(UIPC_MUDA_USE_COREX)` 分支用 `target_compile_definitions(... PUBLIC UIPC_COREX_CUDA10_COMPAT=1)` 注入到全部 cuda 后端、`uipc_core` target，并经 `INTERFACE` / `PUBLIC` 传递给依赖它们的 target。

## 3. 三种允许的分支化模式

### 3.1 单文件 inline 守卫（A-additive / B-mixed-corex 文件）

适合"corex 增量在原文件末尾追加 helper / 修改少数几行"的场景。

```cpp
// type_define.h: corex 路径才需要 force_trivially_* on Eigen::AlignedBox
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
template <typename T, int Dim>
struct force_trivially_destructible<Eigen::AlignedBox<T, Dim>>
{ constexpr static bool value = true; };
// ...
#endif
```

要求：守卫剥掉以后整个文件必须与 `/root/src` 对应文件**字节等价**。

### 3.2 Switcher overlay（B-mixed-corex 大改 / dylib v2/v3 / linear_pcg.cu）

`<file>.cu`（或 `<file>.h` / `<file>.cpp`）改写为只含选择头：

```cpp
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#  include "<file>_corex.cu.inc"   // corex 实现整体放 sidecar
#else
... /root/src 原文 ...             // NVIDIA = /root/src 字节等价
#endif
```

sidecar 命名约定：
- 头文件 → `<file>_corex.h`
- 翻译单元 → `<file>_corex.cu.inc` / `<file>_corex.cpp.inc`

CMake 必须将 `_corex.*` 文件从编译列表里 EXCLUDE（它们仅作为 `#include` 的 inline 内容存在）：

```cmake
list(FILTER SOURCES EXCLUDE REGEX "_corex\\.(cu|h|hpp|cpp|inl)$")
```

### 3.3 CMake-level 文件过滤（NVIDIA-only 或 corex-only 的整组文件）

部分文件不是兼容差异而是平台独占功能（NVIDIA 的 `finite_element_external_force_*`、corex 的 `algorithm/corex_matrix_converter_kernels.*`），用 CMake 排除：

```cmake
if(UIPC_MUDA_USE_COREX)
    list(FILTER SOURCES EXCLUDE REGEX "/finite_element/finite_element_external_force_(manager|reporter)\\.(cu|h)$")
    list(FILTER SOURCES EXCLUDE REGEX "/finite_element/finite_element_external_vertex_force\\.cu$")
endif()

if(NOT UIPC_MUDA_USE_COREX)
    list(FILTER SOURCES EXCLUDE REGEX "/algorithm/corex_matrix_converter_kernels\\.(cu|h)$")
endif()
```

## 4. 禁止事项

- ❌ 不允许直接在源码出现 `#ifdef __CUDACC_VER_MAJOR__ < 11` 之类的隐式平台判定 —— 一律改用 `UIPC_COREX_CUDA10_COMPAT`。
- ❌ 不允许在 corex 路径硬塞 `MUDA_HOST MUDA_DEVICE` 替代 `MUDA_GENERIC` —— [`external/muda/src/muda/muda_def.h`](../external/muda/src/muda/muda_def.h) 已经在 corex 下自动展开，硬替会污染 NVIDIA 路径。
- ❌ 不允许在 fork 自加的 `class` 上引入 `[[nodiscard]]` 而不一并改 `/root/src` —— 否则 §3.1 的"剥守卫后字节等价"会失败。
- ❌ 不允许在 fork 删除 `/root/src` 已有文件 —— 改为加 corex 路径过滤（§3.3）。

## 5. 验证

任何修改 cuda 后端 / `uipc_core` 的 PR **必须**通过：

```bash
bash tools/audit/verify_nvidia_branch_equiv.sh
# 期望: Summary: ok=N fail=0 missing=0
```

实现：[tools/audit/extract_nvidia_branch.py](../tools/audit/extract_nvidia_branch.py) 模拟 `UIPC_COREX_CUDA10_COMPAT` 未定义后的预处理输出，与 `/root/src/<同路径>` 做 `diff -q`。

被纳入 byte_strict 校验的文件清单见 [verify_nvidia_branch_equiv.sh](../tools/audit/verify_nvidia_branch_equiv.sh) 顶部的 `FILES=()` 数组，截至 2026-04-18 共 22 个文件，全部 ok。

> 注：cuda 后端实现层（67 个 `.cu/.h`）暂未纳入 byte_strict 校验，按 fork HEAD 共用方案运行（详见 audit 报告 §12.4）。新增此类文件时**应优先**使用 §3.1/3.2 的 switcher 模式，并把文件名加进 `FILES=()`。

## 6. 升级流程：当 `/root/src` 上游有新版本

1. 用 `diff -rq /root/src /root/libuipc/src` 列出全部差异。
2. 对每个差异点：
   - 是 corex-helper → §3.1 inline 守卫，加进 `FILES=()`。
   - 是结构性差异（field/signature/算法分歧）→ §3.2 switcher overlay，加进 `FILES=()`。
   - 是平台独占文件 → §3.3 CMake 过滤。
3. 跑 `bash tools/audit/verify_nvidia_branch_equiv.sh`，确保 ok 数提高、fail=0。
4. 跑 corex 冒烟（domino 30 frames + scene_surface_*.obj 完整写出）。
5. 拿 fork 去 NVIDIA 机器跑同等的 hello_affine_body / wrecking_ball / corex_demo。

## 7. 历史背景

详细工作记录见 [corex-compat-audit-2026-04-17.md](corex-compat-audit-2026-04-17.md)，特别是：

- §1–§9：原始 167 个差异文件的分类与一期施工。
- §12：2026-04-18 的二次施工（NVIDIA buildable + 22 文件 byte_strict）。
