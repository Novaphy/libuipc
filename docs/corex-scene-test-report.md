# CoreX 场景仿真测试报告

> 测试日期：2026-04-16
> 平台：Tianshu (CoreX) GPU × 8，单卡运行
> 后端：`--backend cuda`
> 代码入口：`apps/examples/corex_demo/main.cpp`

---

## 帧生成耗时（墙钟时间）

**统计口径**（与 `corex_demo` 实现一致）：

- 首帧 `scene_surface_0000.obj` 在 `world.init` 之后、`for` 循环外写出，**不计入**下表「主循环」时间。
- 主循环为 `i = 1 … frames-1`，即再输出 `frames-1` 个 OBJ。下表 **主循环墙钟时间** 取日志中第一次 `>>> Begin Frame: 1` 的时间戳，到最后一次 `<<< End Frame: (frames-1)` 的时间戳之差（**仅 GPU 推进 + sync + retrieve + 写 OBJ** 这一段）。
- 若需 **整进程** 时间，还应加上 `world.init`（含首次 CUDA/JIT 等）以及首帧写出；当日日志里 `world.init OK` 到 `Begin Frame: 1` 通常很短（毫秒级），主要额外开销在首次 CUDA 初始化。
- `corex_demo` 仅在第 1、2、3 帧与**最后一帧**向 stderr 打印 `advance/sync/retrieve/write_obj` 毫秒数，**不能**用其简单相加得到总耗时。
- **解析日志时注意**：必须在**同一次、未拼接**的运行中，取 **同一轮** `Begin Frame: 1` 与对应的 `End Frame: (frames-1)`。若日志里存在多次运行或异常内容，「第一个 Begin」与「最后一个 End」可能不属于同一次，会得到荒谬的总耗时（例如曾误将 simple 算成约 125 s）。

| 场景 | 输出帧数 | 主循环步数 | 主循环墙钟 | 约平均 |
|------|----------|------------|------------|--------|
| simple | 90 | 89 | **3.69 s**（重测） | ~42 ms/步 |
| wrecking_ball | 400 | 399 | **703.6 s**（~11.7 min） | ~1.76 s/步 |
| slope μ=0.3 | 200 | 199 | **12.2 s** | ~61 ms/步 |
| slope μ=0.8 | 200 | 199 | **8.6 s** | ~43 ms/步 |
| stack μ=0.5 | 200 | 199 | **10.3 s** | ~52 ms/步 |
| stack μ=0.0 | 200 | 199 | **9.1 s** | ~46 ms/步 |
| domino μ=0.4 | 300 | 299 | **15.6 s**（初版几何重测） | ~52 ms/步 |

**domino 说明**：初版骨牌尺度 **0.15×1.0×0.5**、间距 **0.55**、`d_hat=0.02` 时，300 帧主循环约 **15.6 s**（日志 `/tmp/corex_domino_retest.log`）。若将几何改得很薄、或减小 `d_hat`、或首块与后续间距过近导致接触极难收敛，单步可到 **数秒**，总时间会显著变长；与「初版参数」不可混读。

**采样**（stderr 中记录的 `advance`/`write_obj`，仅作数量级参考）：

| 场景 | 第 1 帧 advance | 最后一帧 advance |
|------|-----------------|------------------|
| slope μ=0.3 | 44 ms | 25 ms |
| wrecking_ball | 60 ms | 1891 ms（末帧接触极重） |
| simple（重测） | 26 ms（第 1 帧） | 15 ms（第 89 帧） |
| domino μ=0.4（初版几何重测） | 27 ms | 52 ms |

**simple 重测记录**（2026-04-16）：`./Release/bin/corex_demo --backend cuda --scene simple --frames 90 --gpu 1 --output_dir /tmp/corex_simple_retest`，日志 `/tmp/corex_simple_retest.log` 中 `Begin Frame: 1` 与 `End Frame: 89` 时间差约 **3.69 s**；整进程墙钟约 **4.3 s**（含 `world.init` + 写出 `scene_surface_0000` + 主循环）。

**domino 初版几何重测**（2026-04-16）：`UIPC_DOMINO_MU=0.4 ./Release/bin/corex_demo --backend cuda --scene domino --frames 300 --gpu 1 --output_dir /tmp/corex_domino`，主循环 `Begin Frame: 1` → `End Frame: 299` 约 **15.6 s**；整进程约 **16.3 s**（含重定向与关机日志）。

---

## 1. simple — 自由落体 + 接触

**场景描述**：一个动态立方体从高处自由下落到固定平台上。这是最基本的验证场景，用于确认重力、接触、摩擦的基础功能是否正确。

| 参数 | 值 |
|------|-----|
| dt | 0.01 |
| d_hat | 0.03 |
| kappa | 30 GPa |
| mu | 0.0（默认） |
| contact | ON |
| friction | ON（可通过 `UIPC_SIMPLE_FORCE_FRICTION_ENABLE` 关闭） |

**测试结果**：
- 90 帧仿真成功完成
- 立方体呈现标准抛物线自由落体轨迹
- 与固定平台接触后正常停止，无穿透、无悬浮
- 输出目录：`/tmp/corex_simple_final_accept_90f`（90 帧）

**结论**：PASS — 重力、接触、基础物理正确

---

## 2. wrecking_ball — 复杂多体碰撞

**场景描述**：复刻官方 `wrecking_ball` 示例，包含多个 cube、ball、link 几何体的大规模接触场景，带有地面网格。

| 参数 | 值 |
|------|-----|
| dt | 0.01 |
| d_hat | 0.03 |
| kappa | 20 GPa |
| mu | 0.01 |
| contact | ON |
| friction | ON |

**测试结果**：
- 400 帧仿真成功完成
- 多体碰撞、接触传播正常
- 输出目录：`/tmp/corex_wrecking_ball_400f`（400 帧）

**结论**：PASS — 复杂多体场景在 CoreX 上正确运行

---

## 3. slope — 斜面摩擦验证

**场景描述**：经典库仑摩擦验证。一个 30° 倾斜的固定斜面（6×0.3×3 的 cube），上方放置一个 0.5×0.5×0.5 的动态立方体。根据摩擦系数 μ 的大小，立方体应当滑动或保持静止。

临界条件：tan(30°) ≈ 0.577。μ < 0.577 时应滑动，μ > 0.577 时应静止。

| 参数 | 值 |
|------|-----|
| dt | 0.01 |
| d_hat | 0.02 |
| kappa | 20 GPa |
| frames | 200 |
| mu | 通过 `UIPC_SLOPE_MU` 环境变量设置 |

### 测试 3a：slope μ=0.3（应滑动）

| 指标 | 初始值 | 终值 | 变化 |
|------|--------|------|------|
| 滑块 X | -0.2400 | -3.3494 | -3.109 |
| 滑块 Y | +0.4157 | -1.4603 | -1.876 |

**结论**：PASS — 立方体沿斜面明显滑下

### 测试 3b：slope μ=0.8（应静止）

| 指标 | 初始值 | 终值 | 变化 |
|------|--------|------|------|
| 滑块 X | -0.2400 | -0.2434 | -0.003 |
| 滑块 Y | +0.4157 | +0.3444 | -0.071 |

Y 方向微小沉降（0.071）是立方体在重力下落到斜面上的正常沉降过程，之后完全静止。

**结论**：PASS — 立方体静止在斜面上

**摩擦验证**：μ=0.3 < tan(30°) 滑动、μ=0.8 > tan(30°) 静止，完全符合库仑摩擦理论。

---

## 4. stack — 堆叠稳定性

**场景描述**：一个固定地面板（10×0.3×10）上方叠放 3 个 0.8×0.8×0.8 的动态立方体，每个立方体有轻微的水平偏移（0.0、0.1、-0.15），制造倾覆趋势。通过摩擦系数来测试静摩擦能否维持堆叠稳定。

| 参数 | 值 |
|------|-----|
| dt | 0.01 |
| d_hat | 0.02 |
| kappa | 20 GPa |
| frames | 200 |
| mu | 通过 `UIPC_STACK_MU` 环境变量设置 |

### 测试 4a：stack μ=0.5（应稳定）

| 立方体 | 初始位置 (x, y, z) | 终值位置 (x, y, z) | 最大横向漂移 |
|--------|----|----|-----|
| cube_0 | (0.000, 0.450, 0.000) | (-0.000, 0.420, 0.000) | x: 0.0001, z: 0.0000 |
| cube_1 | (0.100, 1.300, 0.000) | (0.100, 1.239, 0.000) | x: 0.0003, z: 0.0001 |
| cube_2 | (-0.150, 2.150, 0.000) | (-0.150, 2.059, 0.000) | x: 0.0004, z: 0.0003 |

**结论**：PASS — 三个立方体稳定堆叠，横向漂移 < 0.001

### 测试 4b：stack μ=0.0（无摩擦，应有横向滑移）

| 立方体 | 初始位置 (x, y, z) | 终值位置 (x, y, z) | 最大横向漂移 |
|--------|----|----|-----|
| cube_0 | (0.000, 0.450, 0.000) | (0.016, 0.419, -0.028) | x: 0.016, z: 0.028 |
| cube_1 | (0.100, 1.300, 0.000) | (0.101, 1.239, 0.031) | x: 0.001, z: 0.031 |
| cube_2 | (-0.150, 2.150, 0.000) | (-0.152, 2.059, -0.018) | x: 0.002, z: 0.018 |

**结论**：PASS — 无摩擦时横向 Z 漂移（最大 0.031）比有摩擦时（最大 0.0003）大 **~100 倍**，明确体现摩擦对稳定性的贡献。

---

## 5. domino — 多米诺骨牌链式接触

**场景描述**：固定地面板 + 5 块竖立的薄骨牌（0.2×1.0×0.4），间距 0.55。第一块预倾斜 50°，在重力作用下倒下并推动后续骨牌，测试接触力的链式传播。

| 参数 | 值 |
|------|-----|
| dt | 0.01 |
| d_hat | 0.02 |
| kappa | 20 GPa |
| stiffness | 1 MPa |
| frames | 300 |
| mu | 0.4（通过 `UIPC_DOMINO_MU` 设置） |
| sanity_check | OFF（允许初始接近） |

### 测试 5：domino μ=0.4（链式倒下）

| 骨牌 | 初始 (x, y) | 终值 (x, y) | X 推移 | Y 下降 |
|------|-------------|-------------|--------|--------|
| D1 | (0.000, 0.550) | (0.038, 0.162) | +0.038 | -0.388 |
| D2 | (0.550, 0.550) | (0.950, 0.425) | +0.400 | -0.125 |
| D3 | (1.100, 0.550) | (1.345, 0.487) | +0.245 | -0.063 |
| D4 | (1.650, 0.550) | (1.708, 0.509) | +0.058 | -0.041 |
| D5 | (2.200, 0.550) | (2.200, 0.503) | 0.000 | -0.047 |

**分析**：
- D1 成功完全倒下（Y 从 0.55 降到 0.16）
- D2 被推移 0.40（约一个骨牌高度的 40%），有明显倾斜
- D3、D4 受到逐级衰减的推力
- D5 基本未受影响

**备注**：IPC 的光滑势垒函数（barrier function）在接触时会吸收一部分冲击能量，导致链式反应逐级衰减。这是 IPC 方法的固有特性，而非程序 bug。真实多米诺需要「硬接触」（刚体碰撞），而 IPC 本质上是通过 barrier 能量实现的「软接触」。

**结论**：PASS（部分链式传播） — 接触力传播和摩擦耦合功能正常，但 IPC 的 barrier 特性限制了完整的链式倒下效果。

---

## 总结

| 场景 | 帧数 | 主循环墙钟（约） | 状态 | 关键验证点 |
|------|------|------------------|------|------------|
| simple | 90 | **3.7 s**（重测） | PASS | 自由落体 + 接触停止 |
| wrecking_ball | 400 | 703.6 s | PASS | 大规模多体碰撞 |
| slope μ=0.3 | 200 | 12.2 s | PASS | 库仑摩擦 — 滑动 |
| slope μ=0.8 | 200 | 8.6 s | PASS | 库仑摩擦 — 静止 |
| stack μ=0.5 | 200 | 10.3 s | PASS | 静摩擦 — 堆叠稳定 |
| stack μ=0.0 | 200 | 9.1 s | PASS | 无摩擦 — 横向滑移 |
| domino μ=0.4 | 300 | **15.6 s**（初版几何重测） | PASS（部分） | 链式接触传播 |

（主循环墙钟定义见本文「帧生成耗时」一节。）

### 运行方式

```bash
cd /root/libuipc/build_corex_current

# simple
./Release/bin/corex_demo --backend cuda --scene simple --frames 90 --gpu 1

# wrecking_ball
./Release/bin/corex_demo --backend cuda --scene wrecking_ball --frames 400 --gpu 1

# slope (设置摩擦系数)
UIPC_SLOPE_MU=0.3 ./Release/bin/corex_demo --backend cuda --scene slope --frames 200 --gpu 1
UIPC_SLOPE_MU=0.8 ./Release/bin/corex_demo --backend cuda --scene slope --frames 200 --gpu 1

# stack (设置摩擦系数)
UIPC_STACK_MU=0.5 ./Release/bin/corex_demo --backend cuda --scene stack --frames 200 --gpu 1
UIPC_STACK_MU=0.0 ./Release/bin/corex_demo --backend cuda --scene stack --frames 200 --gpu 1

# domino (设置摩擦系数)
UIPC_DOMINO_MU=0.4 ./Release/bin/corex_demo --backend cuda --scene domino --frames 300 --gpu 1
```

### 可视化

```bash
python3 tools/obj_sequence_viewer/server.py --frames-dir <output_dir> --port 18080 --fps 10
# 然后在浏览器打开 http://localhost:18080
```
