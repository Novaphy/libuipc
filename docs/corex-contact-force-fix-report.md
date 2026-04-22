# libuipc CoreX/天数 GPU 适配 — 接触力计算修复与 FPU 精度问题工作总结

## 一、项目背景

本轮工作衔接上一轮的碰撞检测与矩阵转换器修复（详见 `corex-collision-contact-fix-report.md`），在碰撞检测（BVH）、轨迹过滤器、矩阵格式转换等基础设施已可用的基础上，**恢复 IPC 接触力的梯度/Hessian 组装**（上一轮暂写零），并修复 CoreX FPU 精度问题导致的仿真物理错误。

上一轮工作结论：碰撞检测正常（PP/PE/EE 对正确检出），但接触力梯度/Hessian 全部写零，物理接触力未生效。

本轮目标：恢复完整的 IPC barrier 梯度/Hessian 计算，使仿真中的接触力正确生效，物体之间无穿透。

---

## 二、场景配置调整

### 2.1 恢复正常初始位置

将上方四面体的初始偏移从 `Vector3::UnitY() * 1.005`（紧贴）改为 `Vector3{0.6, 1.5, 0.0}`，使上方四面体有足够的自由落体空间，便于观察完整的下落→接触→后续行为。

### 2.2 运行参数

```cpp
config["dt"]     = 0.01_s;
config["gravity"] = Vector3{0, -9.8, 0};
config["contact"]["enable"]             = 1;
config["contact"]["friction"]["enable"] = 0;
config["contact"]["d_hat"]              = 0.01;
config["contact_tabular"].default_model(0.5, 1.0_GPa);
```

运行帧数：120 帧。

---

## 三、问题一：IPC 接触力 kernel 从写零恢复为完整计算

### 现象

上一轮中，PP/PE/EE/PT 四个接触力 kernel 中的 barrier gradient/Hessian 因 `muda::Launch().apply()` 复杂 lambda 在 CoreX 上挂死，被替换为写零。接触力未生效，物体直接穿过。

### 解决方式

将四个接触力计算从 `muda::Launch().apply()` lambda 改写为独立的 `__global__` kernel 函数，使用裸指针/viewer 参数传递数据，完全绕过 device lambda 的代码生成问题：

| Kernel | 功能 |
|--------|------|
| `kernel_PP_contact_assemble` | PP 接触对 barrier 梯度/Hessian 组装 |
| `kernel_PE_contact_assemble` | PE 接触对 barrier 梯度/Hessian 组装 |
| `kernel_EE_contact_assemble` | EE 接触对 barrier 梯度/Hessian 组装 |
| `kernel_PT_contact_assemble` | PT 接触对 barrier 梯度/Hessian 组装 |

所有 kernel 均在 `ipc_simplex_normal_contact.cu` 中定义（已有 `.cu` 文件，避免 CoreX 新文件注册失败问题）。

---

## 四、问题二：距离梯度/Hessian 解析函数在 CoreX 上产生错误值

### 现象

恢复完整计算后，Newton 求解器不收敛，梯度值异常（H_max 达到 6.8e+38），仿真 diverge。

### 根因

CoreX FPU 对 double 的有效精度约为 23 位尾数（与 float 相当），导致 libuipc 中以下解析距离导数函数产生灾难性精度丢失：

- `point_edge_distance2_gradient()`
- `point_edge_distance2_hessian()`
- `point_point_distance2_gradient()`
- `point_point_distance2_hessian()`
- `point_triangle_distance2_gradient()` / `_hessian()`
- `edge_edge_distance2_gradient()` / `_hessian()`

这些函数内部涉及大量中间变量相减、相除，在 float 级精度下发生灾难性抵消。

### 解决方式

在 `codim_ipc_simplex_normal_contact_function.h` 中添加 `corex_numgrad` 命名空间，对 PT/EE/PE/PP 四种距离类型实现**数值微分**替代解析导数：

| 函数 | 功能 | eps |
|------|------|-----|
| `PT_distance2_numgrad` | PT 距离平方梯度（中心差分） | 1e-4 |
| `PT_distance2_numhess` | PT 距离平方 Hessian（对梯度再做中心差分） | 1e-3 |
| `EE_distance2_numgrad` | EE 距离平方梯度 | 1e-4 |
| `EE_distance2_numhess` | EE 距离平方 Hessian | 1e-3 |
| `PE_distance2_numgrad` | PE 距离平方梯度 | 1e-4 |
| `PE_distance2_numhess` | PE 距离平方 Hessian | 1e-3 |
| `PP_distance2_numgrad` | PP 距离平方梯度 | 1e-4 |
| `PP_distance2_numhess` | PP 距离平方 Hessian | 1e-3 |

原理：标量距离函数 `point_*_distance2()` 只涉及简单的加减乘运算，在 CoreX 上结果正确。对其做有限差分得到的梯度和 Hessian 精度足够（约 4-6 位有效数字），远好于解析公式的灾难性抵消。

---

## 五、问题三：`KappaBarrier` 系列函数在 CoreX 上产生错误值

### 现象

即使距离梯度/Hessian 修正后，barrier 梯度 `dKappaBarrierdD` 在 CoreX 上返回 ~-1.68e-5，而正确值应为 ~-0.2，差了约 10000 倍。导致接触力约等于零，物体直接穿透。

### 根因分析

IPC barrier 能量公式为：

```
B(D) = -κ · (D - ξ² - U)² · log((D - ξ²) / U)
```

其中 `U = (d_hat + ξ)²`。原始实现中 `D - ξ²` 和 `U` 的差值极小（约 1e-5 量级），在 CoreX 的 float 级精度下：

1. **`std::log()` / `logf()` 在 CoreX 上返回错误值** — 对接近 1 的参数（如 `log(0.99)`）精度严重不足
2. **小双精度数减法** — `D_c - upper` 应为 ~-8e-6 的值，CoreX 返回 -2.0（完全错误）
3. **小双精度数除法** — `D/U`（~1e-4 / ~1e-4）的结果不可靠
4. **函数调用 vs 内联** — `KappaBarrier` 作为函数调用时被 CoreX 编译器错误编译，但同样的代码内联时结果不同

### 解决方式

#### 5.1 `KappaBarrier`：比值公式 + 内联 safe_log

改用比值 `t = (D - ξ²) / U`（避免先算 `D - ξ² - U` 再取 log），并用手工实现的 `safe_log` 替代 `std::log`：

```cpp
// codim_ipc_contact.inl — CoreX 路径
T U = (dHat + xi) * (dHat + xi);
T xi2 = xi * xi;
T t = (D_in - xi2) / U;
// Clamp t to valid barrier range (0, 1)
if(t < T(1e-3)) t = T(1e-3);
if(t > T(0.999)) t = T(0.999);
T tm1 = t - T(1);
// Inline atanh-based log
T u = (t - T(1)) / (t + T(1));
T u2 = u * u;
T logsum = u;
T uk = u;
uk = uk * u2; logsum = logsum + uk / T(3);
uk = uk * u2; logsum = logsum + uk / T(5);
// ... (11 terms)
T log_t = T(2) * logsum;
R = -kappa * (U * U) * (tm1 * tm1) * log_t;
```

`safe_log` 使用 `log(x) = 2·atanh((x-1)/(x+1))` 的 Taylor 展开（8-11 项），配合范围归约 `x/2^n`，精度在 float 级下足够。

#### 5.2 `dKappaBarrierdD`：解析导数公式

对比值公式做解析求导，避免对 `KappaBarrier` 做数值微分（数值微分在 CoreX 上因除法精度不足而失败）：

```
dB/dD = -κ · U · (t-1) · [2·log(t) + (t-1)/t]
```

其中 `t = D_c / U`，`log(t)` 使用 `safe_log`。

#### 5.3 `ddKappaBarrierddD`：对解析一阶导数做数值二阶导数

```cpp
T eps = D_in * T(5e-2);
if(eps < T(1e-8)) eps = T(1e-8);
T gp, gm;
dKappaBarrierdD(gp, kappa, D_in + eps, dHat, xi);
dKappaBarrierdD(gm, kappa, D_in - eps, dHat, xi);
R = (gp - gm) / (T(2) * eps);
```

#### 5.4 `clamp_D` 辅助函数

将输入距离 D 钳制到有效范围 `[ξ² + eps, (d_hat + ξ)²]`，防止 CoreX FPU 产生超出物理范围的距离值：

```cpp
T clamp_D(T D_raw, T dHat, T xi) {
    T upper = (dHat + xi) * (dHat + xi);
    T eps   = upper * T(1e-3);
    T D = D_raw;
    if(D < xi*xi + eps) D = xi*xi + eps;
    if(D > upper)        D = upper * T(0.999);
    return D;
}
```

---

## 六、问题四：`make_spd` 中 `evd_jacobi` 在 CoreX 上返回垃圾特征值

### 现象

`make_spd(H)` 调用后 `H_max` 达到 6.8e+38（正确值应为 ~1e-5），导致 PCG 发散。

### 根因

`make_spd` 原实现使用 Jacobi 迭代法计算特征值分解（`evd_jacobi`），该算法在 CoreX 上由于 FPU 精度问题产生完全错误的特征值，进而将矩阵"投影"到错误的方向。

### 解决方式

替换为基于 **Gershgorin 圆盘定理的对角位移法**：

```cpp
// make_spd.h — CoreX 路径
Float min_gershgorin = H(0, 0);
for(int i = 0; i < N; ++i) {
    Float off_diag_sum = 0;
    for(int j = 0; j < N; ++j)
        if(j != i) off_diag_sum += abs(H(i,j));
    Float lower = H(i,i) - off_diag_sum;
    if(lower < min_gershgorin) min_gershgorin = lower;
}
if(min_gershgorin < 1e-10) {
    Float shift = -min_gershgorin + 1e-10;
    for(int i = 0; i < N; ++i) H(i,i) += shift;
}
```

原理：Gershgorin 定理保证所有特征值落在 `[H(i,i) - Σ|H(i,j)|, H(i,i) + Σ|H(i,j)|]` 范围内。如果最小下界为负，则添加对角位移使所有特征值 ≥ 1e-10。

与原 EVD 方法的差异：原方法会将负特征值截断为 0 然后重构矩阵，Gershgorin 方法只做对角位移。两种方法都保证正定性，但 Gershgorin 更保守（shift 可能偏大），不过计算稳定且完全不依赖浮点迭代。

---

## 七、问题五：barrier 梯度/Hessian 完整组装路径修复

### 变更内容

在 `codim_ipc_simplex_normal_contact_function.h` 中，对四种接触类型的 `*_barrier_gradient` 和 `*_barrier_gradient_hessian` 函数添加 CoreX 路径：

#### 梯度函数（`*_barrier_gradient`）

CoreX 路径使用 `corex_numgrad::*_distance2_numgrad` 计算距离梯度，然后乘以 `dKappaBarrierdD`：

```cpp
// PE_barrier_gradient — CoreX 路径
Float D;
Vector9 GradD;
corex_numgrad::PE_distance2_numgrad(flag, P, E0, E1, D, GradD);
Float dBdD;
dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);
G = dBdD * GradD;
```

PP、PT、EE 同理。

#### Hessian 函数（`*_barrier_gradient_hessian`）

CoreX 路径使用**对 barrier 梯度函数做数值微分**来计算完整 barrier Hessian，绕过了 `ddKappaBarrierddD` 和解析距离 Hessian 的链式法则（两者在 CoreX 上均不可靠）：

```cpp
// PE_barrier_gradient_hessian — CoreX 路径
PE_barrier_gradient(G, flag, kappa, d_hat, thickness, P, E0, E1);

constexpr Float eps = 1e-3;
Vector3 verts[3] = {P, E0, E1};
for(int j = 0; j < 9; ++j) {
    int vj = j / 3, dj = j % 3;
    Vector3 vp[3] = {verts[0], verts[1], verts[2]};
    Vector3 vm[3] = {verts[0], verts[1], verts[2]};
    vp[vj](dj) += eps;
    vm[vj](dj) -= eps;
    Vector9 Gp, Gm;
    PE_barrier_gradient(Gp, flag, kappa, d_hat, thickness, vp[0], vp[1], vp[2]);
    PE_barrier_gradient(Gm, flag, kappa, d_hat, thickness, vm[0], vm[1], vm[2]);
    H.col(j) = (Gp - Gm) / (2.0 * eps);
}
H = (H + H.transpose()) * 0.5;  // 对称化
```

PP (6×6)、PT (12×12)、EE (12×12，含 mollifier) 同理。

---

## 八、当前仿真状态与遗留问题

### 8.1 仿真结果

120 帧仿真在 ~6.6 秒内完成，Newton 收敛正常（1-22 次迭代），H_max 量级正确（~1e-5），未出现 diverge 或挂死。

**但物理结果仍然不正确**：上方四面体穿透了下方四面体。

| 帧 | 上方四面体顶点 Y | 上方四面体底面 Y (平均) | 期望 |
|------|-----------------|----------------------|------|
| 0 | 2.50 | 1.50 | 初始位置 |
| 50 | 1.27 | 0.27 | 应在 ~1.0 处接触停止 |
| 60 | 1.00 | 0.00 | 已经完全穿透 |
| 119 | 1.01 | 0.00 | 穿透后落到地面位置 |

下方四面体占据 Y=[0, 1]，上方四面体底面在 frame 60 已落至 Y≈0，**直接穿过了下方四面体**，说明接触力几乎为零。

### 8.2 根因分析

**`dKappaBarrierdD` 返回值仍然比正确值小约 10000 倍**。尽管已经改为解析公式，但公式中依赖的关键运算 `t = D_c / U`（其中 D_c ~8e-5, U ~1e-4）在 CoreX FPU 上返回不正确的结果。

整个修复过程中，每次尝试的方案最终都依赖于同一个 CoreX 无法正确执行的运算——两个小 double 数的除法。具体的失败链路为：

```
barrier 梯度 dB/dD = -κ · U · (t-1) · [2·log(t) + (t-1)/t]
                            ↑
                        t = D_c / U  ← CoreX FPU 在此处返回错误结果
                        D_c ≈ 8e-5
                        U   ≈ 1e-4
```

### 8.3 Newton 为什么"收敛"了但物理错误

Newton 收敛≠物理正确。当接触力近乎为零时，系统只剩重力 + 弹性势能，这是一个简单的自由落体问题，Newton 当然快速收敛。但由于缺乏足够的接触力，物体直接穿过了对方。

---

## 九、全部修改文件汇总（本轮）

```
apps/examples/corex_demo/main.cpp
    — 恢复上方四面体初始位置为 {0.6, 1.5, 0.0}
    — 添加 CLI 参数解析（--frames, --output_dir, --scene, --backend）
    — 添加逐帧计时日志

src/backends/cuda/contact_system/contact_models/ipc_simplex_normal_contact.cu
    — 将 PP/PE/EE/PT 从 Launch lambda 改写为 __global__ kernel
    — 恢复完整 barrier gradient/Hessian 计算（替代写零）
    — 恢复 make_spd(H) 调用

src/backends/cuda/contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h
    — 添加 corex_numgrad 命名空间（8 个数值微分函数：4 种距离类型 × 梯度/Hessian）
    — 添加 CoreX 路径的 *_barrier_gradient（使用 numgrad 距离梯度 + 解析 dBdD）
    — 添加 CoreX 路径的 *_barrier_gradient_hessian（对 barrier 梯度做数值微分得 Hessian）

src/backends/cuda/contact_system/contact_models/sym/codim_ipc_contact.inl
    — 添加 corex_barrier_detail 命名空间（clamp_D, safe_log）
    — KappaBarrier：比值公式 + 内联 atanh-based log
    — dKappaBarrierdD：解析导数 -κ·U·(t-1)·[2·log(t)+(t-1)/t]
    — ddKappaBarrierddD：对解析一阶导数做数值二阶导数

src/backends/cuda/utils/make_spd.h
    — CoreX 路径：Gershgorin 对角位移法替代 evd_jacobi

src/backends/cuda/affine_body/abd_linear_subsystem.cu
    — 清理调试日志（corex_diag 数据回读）

src/backends/cuda/affine_body/abd_line_search_reporter.cu
    — CoreX 路径：step_forward 改写为 __global__ kernel
    — CoreX 路径：能量汇总使用 host 端 reduce（kinetic_energy, shape_energy, reporter_energies）
```

---

## 十、当前仿真状态

| 阶段 | 状态 |
|------|------|
| BVH 碰撞检测 | ✅ 正常工作 |
| 轨迹过滤器 AABB | ✅ 正常工作 |
| 接触力能量计算 (`do_compute_energy`) | ✅ Launch lambda 执行完整计算 |
| 接触力梯度/Hessian (`do_assemble`) | ⚠️ kernel 执行、Newton 收敛，但 **dBdD 返回值比正确值小 ~10000 倍**，接触力几乎为零 |
| 矩阵格式转换 (Triplet→BCOO→BSR) | ✅ 正常工作 |
| 线性求解 (PCG) | ✅ 正常收敛 |
| make_spd | ✅ Gershgorin 方法正常工作 |
| **物理正确性** | ❌ **穿透** — 上方四面体直接穿过下方四面体 |

---

## 十一、待解决事项

### 高优先级

1. **修复 `dKappaBarrierdD` 在 CoreX 上的精度问题** — 当前解析公式依赖 `t = D_c / U`（两个 ~1e-4 量级的 double 相除），CoreX FPU 无法正确计算。需要找到不依赖小数除法的计算路径。可能的方案：
   - **方案 A**：完全绕过 dBdD 和 GradD 的分别计算，直接对 barrier 能量关于顶点坐标做数值微分（`G[j] = (B(x+ε·ej) - B(x-ε·ej)) / 2ε`），但需要验证数值精度是否足够
   - **方案 B**：使用缩放技巧将 D 和 U 同时放大到 O(1) 量级再做除法
   - **方案 C**：将 barrier 计算中的关键运算（log、division）拆分为多步并在每步后做精度补偿

### 中优先级

2. **验证 CCD 轨迹过滤器** — `GlobalTrajectoryFilter` 中的 `filter_toi` 始终返回 `alpha=1.0`（不限制步长），可能需要独立验证其正确性

3. **清理残留的 host fallback** — `abd_line_search_reporter.cu` 中的能量汇总仍使用 host 端 cudaMemcpy + CPU reduce

### 低优先级

4. **开启 friction** — 在 contact 完全正常后开启

5. **性能优化** — 数值微分方案（每个 contact pair 需要 2N 次额外的距离计算）带来一定的性能开销，待接触力正确后评估是否需要优化

---

## 十二、核心经验总结（本轮新增）

1. **CoreX FPU 的 double 精度约等于 float** — 对于 IPC 这类依赖 double 精度的算法，几乎所有包含小数减法（D - ξ²）、小数除法（D/U）、接近 1 的 log（log(0.99)）的计算都会产生错误结果。不能简单地用"换一个等价公式"来绕过，因为等价公式最终仍会依赖这些基本运算。

2. **Newton 收敛不等于物理正确** — 当接触力太小时，系统退化为无接触的自由落体，Newton 快速收敛是因为问题变简单了，不是因为接触力在正常工作。需要通过实际的 OBJ 顶点坐标来验证物理正确性。

3. **数值微分在 CoreX 上部分可用** — 对简单函数（距离平方）做数值微分可行，因为函数值计算只涉及加减乘；但对复杂函数（barrier 能量）做数值微分不可行，因为 barrier 函数自身的计算就依赖有问题的运算。

4. **修复需要从最底层的数值运算入手** — 在更高层面（梯度公式、Hessian 公式）做替换，如果底层运算（除法、log）仍然有问题，修复就无法生效。下一步需要找到完全不依赖 `D/U` 除法的 barrier 梯度计算路径。

5. **`evd_jacobi` 在 CoreX 上不可用** — Jacobi 迭代的收敛依赖精确的旋转/消元操作，float 级精度下会产生完全错误的特征值。Gershgorin 对角位移是一个可靠的替代方案。
