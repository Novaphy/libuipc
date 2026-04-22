# CoreX Runtime 异常排查记录（2026-04-14）

## 背景

- 目标：继续推进 `libuipc` 单精度（float）适配，恢复 CoreX 上可运行基线。
- 现象：`corex_demo` 在 `world.init()` 早期卡住；此前日志常表现为 `cudaErrorUnknown (999)`，或在探针阶段直接无返回。
- 约束：不采用主机端回写（host-side writeback）作为长期方案。

## 本次排查范围

本次重点不是接触/求解数值正确性，而是先确认“为什么现在连基本 API/初始化都不稳定”。

主要覆盖：

1. `SimEngine` 构造前后 Runtime 状态探测；
2. 动态库加载与 PMR 同步路径；
3. `AffineBodyDynamics::_build_geometry_on_device` 首次设备分配路径；
4. 独立最小 API 测试与 libuipc 主路径的交叉验证；
5. 设备维度（GPU 0 vs 其他卡）差异验证。

## 关键结论（先说结论）

1. **单精度改动不是独立 API 测试异常的直接原因。**
2. **当前阻塞是设备维度问题：默认物理 GPU 0 异常，GPU 7 正常。**
3. **在 GPU 7 上，最小 probe 与 `corex_demo` 均可通过 `cudaStreamCreate`/`cudaMalloc` 并成功出 OBJ。**
4. **之前“所有 CoreX API 都坏”的结论需要修正为：当前会话默认设备路径异常，不是全局不可用。**

---

## 过程与证据

### 1) PMR 路径排查：不是根因

对以下链路增加边界日志：

- `core/internal/engine.cpp`：`dlopen -> uipc_init_module -> uipc_create_engine`
- `backends/common/module.cpp`：`uipc_init_module` entry/leave
- `backends/cuda/entrance.cpp`：`uipc_create_engine` entry
- `backends/common/sim_engine.cpp`：base ctor entry
- `backends/cuda/engine/sim_engine.cu`：derived ctor entry

结果：

- `dlopen`、PMR 同步、`uipc_create_engine`、基类构造均正常通过；
- 异常发生在 CUDA 派生构造后的 Runtime 调用阶段。

### 2) Runtime 探针分段化后定位

对 `SimEngine` 增加分段 probe（可开关），确认：

- `cudaGetDeviceCount`、`cudaGetDeviceProperties`、`cudaGetDevice` 可返回；
- 在某些运行中 `cudaStreamCreateWithFlags` 可直接卡住；
- 在绕开 stream 探针时，又会推进到首次 `cudaMalloc` 并卡住。

说明：存在“stream/create 路径”和“首次 alloc 路径”双重不稳定，而不是单个 API 永远失败。

### 3) `AffineBodyDynamics` 分配链对齐与回退

当前分支 CoreX 路径有 `corex_raw_resize`（手工管理 `DeviceBuffer` 内存）。为避免单一路径误判，调整为：

- `cudaMalloc` -> `cudaMallocPitch` -> `cudaMalloc3D` 回退链；
- 每步增加 begin/done/failed 日志。

观察到的卡点仍可能出现在第一步 `cudaMalloc`（无返回）。

### 4) 独立 API 测试交叉验证

独立测试 `tools/corex_api_tests/corex_runtime_extended` 与 `tools/corex_cuda_api_probe` 复测时，先后出现：

- 卡在 `cudaStreamCreateWithFlags`；
- 或跳过 stream 后卡在 `cudaMalloc`。

这与 libuipc 主路径现象一致，证明问题不局限于 libuipc 业务逻辑。

### 5) 设备维度验证（关键转折）

在最小 probe 上对比：

- 默认设备路径：卡在 `cudaStreamCreate` 或 `cudaMalloc`；
- `CUDA_VISIBLE_DEVICES=7`：`cudaStreamCreate`、`cudaMalloc`、Runtime smoke 全通过（仅保留历史已知 WARN：如 `cublasDdot NOT_SUPPORTED`、`cusparse` 兼容告警等）。

随后在 libuipc 上验证：

- `CUDA_VISIBLE_DEVICES=7 UIPC_COREX_PROBE_STEPS=get_device ./Release/bin/corex_demo ...`
- `world.init OK`，成功输出 `scene_surface_0000.obj`。

结论：**默认 GPU 0 路径异常，非全卡、非全局 Runtime 不可用。**

---

## 本次代码改动（用于定位与恢复基线）

### A. 诊断与边界日志

- `src/core/core/internal/engine.cpp`
  - 增加 `load_module` / `dlopen` / `uipc_init_module` / `uipc_create_engine` 边界日志。
- `src/backends/common/module.cpp`
  - 增加 backend module initializer enter/leave 日志。
- `src/backends/cuda/entrance.cpp`
  - 增加 `uipc_create_engine` entry 日志。
- `src/backends/common/sim_engine.cpp`
  - 增加 base ctor entry 日志。
- `src/backends/cuda/engine/sim_engine.cu`
  - 增加分段 runtime probe；
  - 增加 `UIPC_SKIP_CUDA_SET_DEVICE`（诊断开关）；
  - 增加 `UIPC_COREX_WARMUP_ALLOC`（诊断开关）；
  - probe 默认改为仅 `get_device`，其余步骤显式开启。

### B. ABD 分配路径观测增强

- `src/backends/cuda/affine_body/affine_body_dynamics.cu`
  - `corex_raw_resize` 改为 `cudaMalloc -> cudaMallocPitch -> cudaMalloc3D`；
  - 增加详细分配日志（begin/done/failed）。

### C. 示例程序可控选卡（用于绕开异常卡恢复基线）

- `apps/examples/corex_demo/main.cpp`
  - 新增 `--gpu` 参数解析；
  - 支持环境变量 `UIPC_COREX_GPU_DEVICE`；
  - 将设备号写入 **Engine config**（非 Scene config）；
  - 启动日志打印目标设备。

---

## 当前可复现实验命令

### 1) 最小 probe（设备 7，验证 Runtime 正常）

```bash
cd /root/libuipc/tools/corex_cuda_api_probe
CUDA_VISIBLE_DEVICES=7 ./build_match_history/corex_cuda_api_probe
```

### 2) `corex_demo`（设备 7，验证可出帧）

```bash
cd /root/libuipc/build_corex_current
./Release/bin/corex_demo --backend cuda --scene simple --frames 1 --gpu 7 --output_dir /tmp/corex_demo_gpu7_cli3
```

成功标志：

- 日志包含 `world.init OK`；
- 输出目录包含 `scene_surface_0000.obj`。

---

## 后续建议（按优先级）

1. **先固定健康设备基线（如 GPU 7）继续单精度正确性收敛**，避免被异常卡阻塞。
2. 对 GPU 0 做独立 runtime 体检（仅 Runtime API，不带业务代码），必要时联系驱动/运维侧确认卡状态。
3. 将示例与回归脚本统一支持 `--gpu`，并在文档明确“指定设备复现”。
4. 数值正确性（contact/CCD/solver）继续在“可运行基线”上推进，避免 runtime 噪声混入结论。

---

## 计划继续推进（同日增补）

在恢复可运行基线后，已继续推进到计划中的 `contact-ccd-audit`：

1. `contact_system/contact_models/sym/codim_ipc_contact.inl`
   - 将 CoreX 数值导数路径的固定 `eps` 下限替换为基于 `std::numeric_limits<T>::epsilon()` 的自适应步长函数（`finite_diff_step`），减少 float 下过小步长引发的数值噪声。
2. `contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h`
   - 将 `kappa / 4.0`、`kappa / 2.0` 改为 `Float(4)`、`Float(2)`，避免不必要的 double 字面量参与运算。

验证结果（`--gpu 7`）：

- 1 帧 smoke：通过，`world.init OK`，成功出 OBJ。
- 60 帧 simple 场景回归：通过（输出目录文件数 62，含 `scene_surface_0000.obj` ~ `scene_surface_0060.obj` + 配置文件），日志连续显示 Newton 迭代收敛，无崩溃/初始化失败。

说明：

- 本轮变更属于“低风险 float 语义收敛”，未引入新的 runtime 回归。
- 下一步继续按计划进入 `solver-mixed-precision`，收敛 PCG/线搜索等阈值与 mixed-precision 残留点。

---

## 计划继续推进（同日增补二）

按原计划继续完成 `solver-mixed-precision` 的一批低风险收敛，并保持“**不使用主机端回写**”约束：

1. `linear_system/linear_pcg.cu`
   - 将 CoreX 分支中的 `update_xr/update_p` 从 D2H/H2D 回写路径恢复为纯 device kernel 更新（`kernel_update_xr/kernel_update_p`）。
2. `line_search/line_searcher.cu`
   - `std::accumulate` 初值从 `0.0` 改为 `Float(0)`，避免 float 路径隐式 double 累加。
3. `finite_element/constitutions/*plastic_discrete_shell_bending_function.h`
   - 对 `plasticity_write_threshold` 与 `dihedral_guard_eps` 增加 float 专用阈值分支（double 保持原值）。
4. `affine_body/*revolute_joint*.cu`
   - 将 `1e-12/1e-24` 级硬编码关节轴/力臂阈值改为 float/double 分支阈值函数，减少 float 下过严判定导致的不稳定。

对照检查（`/root/src`）：

- `line_searcher.cu` 与两份 `*plastic_discrete_shell_bending_function.h` 的差异均为本轮预期的 float 语义收敛。
- `affine_body_revolute_joint.cu` 的新增差异也集中在阈值函数化与 float 分支。
- `affine_body_revolute_joint_external_force*.cu` 在 `/root/src` 无同名对照文件，已在本分支内保持改动局部且可回归验证。

验证方式（按产物而非终端刷屏）：

- 命令：
  - `./Release/bin/corex_demo --backend cuda --scene simple --frames 60 --gpu 7 --output_dir /tmp/corex_demo_simple_gpu7_no_host_writeback`
  - `./Release/bin/corex_demo --backend cuda --scene simple --frames 60 --gpu 7 --output_dir /tmp/corex_demo_simple_gpu7_solvermix2`
- 结果：
  - 两次均成功生成 `scene_surface_0000.obj` ~ `scene_surface_0059.obj`（共 60 帧，连续完整）。
  - 日志尾部均包含 `Wrote OBJ sequence` 与 `frame 59 timings`，未出现初始化失败或中途中断。

---

## 备注

- 本文档聚焦“运行时异常定位与恢复运行基线”，不替代接触/求解精度专项报告。
- 本轮定位结论：**当前主要阻塞已从“代码语义错误”转为“特定设备 runtime 状态异常”。**

## 同日续推进（三）：`simple` 场景穿透（候选全 0）链路修复

在 `simple` 场景继续排查后，确认“穿透”并非单一求解器收敛问题，而是先后叠加了两层问题：

1. **候选生成前置过滤错误（contact mask）**
   - 在 `StacklessBVHSimplexTrajectoryFilter::Impl::detect()` 增加输入诊断后，观察到：
     - `Vs=8, Es=12, Fs=8` 均正常；
     - 顶点 `cid/scid/body` 写入正常；
     - 但 `contact_mask_tabular` 为 `3x3`，且 `row1/row2` 全 0，导致跨体样本 `cid=(1,2)` 的 `contact_allow=0`，候选被全部提前过滤。
   - 该问题由 `corex_demo` 的 simple 场景使用自定义 contact element（`falling/fixed`）触发。

2. **接触提前量/刚度不足导致接触时机偏晚**
   - 将 simple 场景 contact element 统一为 `default_element` 后，`detect`/`filter_active` 恢复非零；
   - 进一步把 simple 场景参数收敛为：
     - `contact/d_hat = 0.05`
     - `default_model(..., resistance = 20.0_GPa)`
   - 40 帧回归（GPU 1）中，`frame 28~36` 的最小间隙保持正值，未再出现第 32 帧起直接穿透。

关键变更文件：

- `apps/examples/corex_demo/main.cpp`
  - simple 场景 contact element 绑定改为 `default_element`；
  - simple 场景 `d_hat` 调整为 `0.05`；
  - simple 场景 default contact resistance 调整为 `20.0_GPa`。
- `src/backends/cuda/collision_detection/filters/stackless_bvh_simplex_trajectory_filter.cu`
  - 增加低频输入诊断日志（`detect_input`）：几何规模、`cid/scid/body` 样本、mask 维度与样本值，用于确认候选清零责任层。

## 同日续推进（四）：`simple` 全功能恢复与“接触后静止”收敛

本轮按 `simple全功能仿真修复` 计划继续推进，目标是恢复 `simple` 的完整功能并避免“接触后直接数值静止”。

### 已恢复/确认项

1. `simple` 场景级别功能恢复
   - `contact/friction/enable = 1`
   - `sanity_check/enable = 1`
   - `contact/d_hat` 进行了阶梯回收测试（`0.05 -> 0.03 -> 0.025 -> 0.02`），最终选择 `0.025` 作为当前折中点。
2. 运行时模块路径修复
   - `apps/examples/corex_demo/main.cpp` 中 `module_dir` 优先指向 `Release/bin`，避免开启 sanity_check 后找不到 `libuipc_sanity_check.so`。
3. 固定体实例标志修复
   - simple 场景 fixed 物体显式设为 `is_dynamic = 0`（避免 fixed/dynamic 混杂状态）。
4. 诊断日志收敛
   - `stackless_bvh_simplex_trajectory_filter.cu` 中移除了 `kernel_filter_active_PT` 的设备端 `printf`；
   - CoreX 过滤阶段 host 诊断日志改为低频采样，避免日志洪泛拖慢仿真。

### 全局默认恢复评估结论

1. `collision_detection/method = info_stackless_bvh`
   - 在当前 CoreX 路径下会导致候选再次退化为全 0，已回退为 `stackless_bvh`。
2. `linear_system/tol_rate = 1e-3`
   - 可运行，保留。
3. `newton/max_iter`
   - 提升到更高上限会显著放大单帧耗时，当前保留 `100` 作为效率/稳定折中。
4. `fused_pcg`
   - 已完成评估；在 simple 可运行，但综合稳定性与可控性，当前默认仍保留 `linear_pcg`。

### 当前 simple 推荐配置（CoreX + float）

- `contact/friction/enable = 1`
- `sanity_check/enable = 1`
- `contact/d_hat = 0.025`（simple override）
- `scene.contact_tabular().default_model(0.0, 30.0_GPa)`（simple）
- `collision_detection/method = stackless_bvh`
- `linear_system/solver = linear_pcg`
- `linear_system/tol_rate = 1e-3`

### 回归结果（GPU 1）

回归命令：

- `./Release/bin/corex_demo --backend cuda --scene simple --frames 60 --gpu 1 --output_dir /tmp/corex_simple_candidate_dhat0025_k30_60f`

结果摘要：

- 60 帧完成（`scene_surface_0000.obj` ~ `scene_surface_0059.obj`）。
- `frame 28~36` 接触窗口内最小 gap 保持为正；全程 `min_gap = 0.002057`，无穿透。
- 检测链路持续非零：`detect` 中 `PT_cands/EE_cands` 非零，`filter_active` 中 `PE` 非零。
- 接触后 `32~59` 帧仍有位姿变化（非完全冻结），但 `40~59` 帧变化幅度已明显减小，说明当前仍存在“后段趋于静止”的残余问题，需在后续继续优化线搜索/接触动力学耦合。

## 同日续推进（五）：按 90 帧标准推进 line-search / ABD / 参数回收

按 `simple_接触后静止后续修复计划` 执行后续三阶段，并统一使用 **90 帧非快速回归**：

### 1) line-search 一致性修正（保持稳定优先）

文件：

- `src/backends/cuda/engine/advance_ipc.cu`

改动：

- 增加 `energy_acceptance_tol(E0)`，将接收条件从硬 `E <= E0` 改为 `E <= E0 + tol`。
- float 路径容忍度收敛为相对 `1e-6` 量级（仅覆盖浮点噪声，不放大真实能量上升）。

说明：

- 试过把 `E0` 与 trial step 的 DCD 上下文强制对齐，但在当前 CoreX 路径会引入明显反效果，已回退该部分，仅保留容忍度修正。

### 2) ABD fixed-dynamic Hessian 修正（最小安全版）

文件：

- `src/backends/cuda/affine_body/abd_linear_subsystem.cu`

改动：

- 在 CoreX `kernel_abd_dytopo_hessians_serial` 与非 CoreX `_assemble_dytopo_effect` 路径中：
  - 对 fixed-dynamic pair 不再“只做 `H12x12.setZero()`”；
  - 先将 `H3x3` 对称化为 `Hsym`，再把 dynamic 侧局部曲率 `JT_H_J(J_dyn^T, Hsym, J_dyn)` 累加到 `diag_hessian(dynamic)`；
  - coupling triplet 仍保持清零，避免破坏 fixed 侧约束语义。

目标：

- 在不放宽 fixed 约束的前提下，给 dynamic 侧保留必要二阶信息，缓解“接触后只剩梯度、曲率过弱”的静止倾向。

### 3) 接触参数回收（90 帧对照）

文件：

- `apps/examples/corex_demo/main.cpp`

90 帧对照结论：

- `d_hat=0.02` 与 `d_hat=0.01` 均可稳定运行；采用 tetra 包含关系代理检查，均未出现“顶点进入对方四面体”。
- `d_hat=0.01 + resistance=20GPa` 的后段姿态变化明显减弱（偏静止）。
- 当前折中选择：`simple` 使用 `d_hat=0.02`，`default_model(0.0, 30.0_GPa)`。

### 4) 90 帧最终回归（当前折中配置）

命令：

- `./Release/bin/corex_demo --backend cuda --scene simple --frames 90 --gpu 1 --output_dir /tmp/corex_simple_final_plan_90f`

结果摘要：

- 90 帧完成（`scene_surface_0000.obj` ~ `scene_surface_0089.obj`）。
- 检测链路持续非零：`detect` 中 `PT/EE` 候选非零，`filter_active` 中 active 非零。
- 日志未出现 `Line Search Exits with Max Iteration`，CCD/CFL 也未出现异常收缩刷屏。
- 接触后 `36~89` 帧 COM 与姿态持续变化（不再是短时间内数值冻结）。
- 使用四面体包含关系代理检查，未观察到互相进入体积内部的帧。

备注：

- `y` 向投影间隙在后段会转负，主要反映物体沿接触面滑落并越过固定体高度，不能单独作为“穿透”判据；已改用体积包含代理校验。
