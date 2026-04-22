# CoreX Runtime Worklog 2026-03-31

## 1. 项目当前阶段

当前 `libuipc` 在 CoreX/TianShu 环境下已经具备以下能力：

- 可以完成 CoreX CUDA backend 编译。
- `corex_demo` 可以正常加载生成的 `.so` 并启动仿真。
- 早期的“首帧卡死在初始化阶段”问题目前没有再次复现。
- `simple` 场景可以跑完多帧并导出 OBJ。

但当前仍然没有恢复到“正常连续运动”的目标状态：

- `simple` 场景下 `scene_surface_0001.obj == scene_surface_0002.obj`，后续帧静止。
- `LinearPCG` 始终显示 `iters=0`。
- 当前更准确的问题表述不是“首帧卡死”，而是“首帧之后求解链没有把位移真正推进起来”。

## 2. 当前已确认的问题收敛

到今天为止，问题已经从“大范围 CoreX 兼容性不确定”收敛到下面这条主链：

1. `ABDLinearSubsystem` 的 shape contribution 在 CoreX 路径里原本被整体清零/跳过。
2. 恢复 `OrthoPotential` 后，装配链已经不再是全零，`norm(b)` 已经能变成非零。
3. 但 `LinearPCG` 仍然在初始化阶段早退，日志表现为：
   - `norm(b) != 0`
   - `norm(r) == 0`
   - `norm(z) == 0`
   - `iters=0`
4. 因此当前最可疑的区域已经收敛为：
   - `PCG` 初始化阶段的 `b -> r`
   - `r -> z` 的预条件器路径
   - `accuracy_statisfied(r)` 与默认/自定义 stream 的可见性问题

## 3. 今天完成的代码调整

### 3.1 恢复 `OrthoPotential` 的 CoreX 可运行路径

文件：

- `src/backends/cuda/affine_body/constitutions/ortho_potential.cu`
- `src/backends/cuda/affine_body/abd_linear_subsystem.cu`

本次调整内容：

- 为 `OrthoPotential` 增加了 CoreX 下的 host fallback。
  - `do_compute_energy()` 改为设备数据拷回 host，CPU 计算后再写回 device。
  - `do_compute_gradient_hessian()` 同样改为 host 侧计算并回写。
- 在 `ABDLinearSubsystem::_assemble_kinetic_shape()` 中，不再继续“CoreX 下跳过全部 constitution”。
- 改为仅对白名单中的 `OrthoPotential`（UID `1ull`）恢复 shape contribution，其它 constitution 暂时继续跳过。

这样做的目的：

- 先恢复 `simple` 场景最关键的一条材料路径。
- 避免一次性把所有 affine constitution 都放开，扩大 CoreX 运行时不兼容面。

当前结果：

- 相关 trace 已确认 `constitution 0 uid=1 begin/end` 实际执行。
- 说明 `OrthoPotential` 已经进入求解链，不再是“完全没跑到”。

### 3.2 补 `muda` 在线性系统里的 CoreX `norm/dot` 运行时 fallback

文件：

- `external/muda/src/muda/ext/linear_system/details/routines/norm.inl`
- `external/muda/src/muda/ext/linear_system/details/routines/dot.inl`

本次调整内容：

- 为 `norm.inl` 增加 host fallback。
  - 处理 CoreX 下 `cublasSnrm2` / `cublasDnrm2` 返回 `CUBLAS_STATUS_NOT_SUPPORTED` 的情况。
  - 回退到 `cudaMemcpy(DeviceToHost) + host 侧求范数`。
- 为 `dot.inl` 与 `norm.inl` 的 host fallback 增加 `sync()`。
  - 避免在 `muda` 自己的 linear-system stream 上的计算尚未完成时，直接把 device 数据拷到 host，读到旧值。

这样做的目的：

- 之前恢复 `OrthoPotential` 后，运行时直接暴露出了 CoreX 的 cuBLAS 不支持问题。
- 不补齐 `dot/norm` fallback，就无法继续往 PCG 真正的数值路径推进。

当前结果：

- 已经不再在 `cublasDnrm2` 处直接崩溃。
- 现在可以继续看到 PCG 的早退行为和更细的数值状态。

### 3.3 增加 `LinearPCG` 诊断并开始清理默认 stream 干扰

文件：

- `src/backends/cuda/linear_system/linear_pcg.cu`

本次调整内容：

- 增加了 `LinearPCG` 的早退日志：
  - `norm(b)`
  - `norm(r)`
  - `norm(z)`
- 将下列操作开始切到 `ctx().stream()`：
  - `x` 清零
  - `r = b`
  - `p = z`
  - `update_xr`
  - `update_p`
- 在 CoreX 下对 `z/p/r/Ap` 的 `resize` 完成后加入一次 `wait_device()`，避免默认 stream 上的 buffer 操作还未可见就进入线性系统 stream。

这样做的目的：

- 当前现象非常像“线性系统装配和 PCG 初始化不在同一条 stream 上”，导致 `b` 已经是非零，但 `r`/`z` 仍被读成零。
- 这类问题在 CoreX 上比 NVIDIA 更容易暴露。

当前结果：

- 修改已编译通过。
- 但最新复跑结果表明，仅修 `LinearPCG` 这一侧还不够，`norm(r)` / `norm(z)` 仍然是零。

### 3.4 修复“每次几乎全量编译”的构建问题

文件：

- `CMakeLists.txt`

本次调整内容：

- 对 CoreX/ILUVATAR 的 CUDA Ninja 规则强制补回 depfile 参数：
  - `-MD -MT <DEP_TARGET> -MF <DEP_FILE>`

额外处理：

- 清理并重建了当前 build 目录中的 `.ninja_deps` 和 `.ninja_log`。

这样做的目的：

- 之前 Ninja 依赖数据库损坏，且 `CUDA_COMPILER__cuda_` 规则没有生成 depfile。
- 导致大量 `.cu.o` 永远处于 `deps ... are missing` 状态，每次构建都像“全量重编”。

当前结果：

- 一次性重建后，第二次相同命令已经可以直接返回：
  - `ninja: no work to do.`
- 当前 build 目录的增量编译能力已经恢复。

## 4. 今天的验证结果

### 4.1 构建结果

- `corex_demo` 可以完成构建。
- 增量编译已恢复正常。

### 4.2 运行结果

测试命令：

```bash
LD_LIBRARY_PATH=.:/usr/local/corex/lib64 ./corex_demo --backend=cuda --scene=simple --frames=3
```

当前关键日志结论：

- `world.init OK`
- `GlobalLinearSystem has 24 DoFs, Unique Triplet Count: 1`
- `LinearPCG: early exit with zero initial rz`
- 首次 `newton_iter=0` 时 `norm(b)=0`
- 后续 `newton_iter=1..3` 时 `norm(b)=3637308.0214742282`
- 但对应的 `norm(r)=0, norm(z)=0`

OBJ 校验结果：

- `scene_surface_0000.obj` 与 `scene_surface_0001.obj` 不同
- `scene_surface_0001.obj == scene_surface_0002.obj`

因此当前结论是：

- 装配出的右端项 `b` 已不再全零。
- 问题已收敛到 `PCG` 初始化/预条件器阶段，而不是 shape contribution 本身。

## 5. 下一步工作方向

下一步不再继续扩展更多 constitution，而是集中推进 `PCG` 初始化链：

1. 继续清理 `GlobalLinearSystem::apply_preconditioner()` 中默认 stream 的 `r -> z` 拷贝。
2. 对 `r = b` 后的真实 device 内容做更直接的 host dump，确认 `r` 是“被拷成零”还是“被后续路径覆盖为零”。
3. 检查 `accuracy_statisfied(r)`、local/global preconditioner 的实现是否存在 CoreX 下的 stream 可见性问题。
4. 若 `r` 与 `z` 确认仍被默认 stream 影响，则将 `GlobalLinearSystem` 线性系统内部这几处 buffer copy/fill 全部显式绑定到 `ctx().stream()`。
5. 待 `PCG` 不再 `iters=0` 后，再回到 OBJ 连续运动验证，并逐步恢复复杂场景及 contact/friction。

## 6. 目前建议的工作优先级

优先级从高到低如下：

1. 先打通 `simple` 场景下 `PCG` 的非零迭代。
2. 再验证 `simple` 场景 OBJ 连续变化。
3. 再考虑恢复更多 affine constitutions。
4. 最后再回接 contact、friction、复杂 scene。

## 7. 当前工作结论

截至 2026-03-31，本轮工作的价值主要有两点：

- 构建侧已经恢复可持续迭代，不再被“每次几乎全量重编”拖慢。
- 运行侧已经把问题从“CoreX 下整体不可控”收敛成“PCG 初始化/预条件器链条的具体异常”，后续排查方向已经明显变窄。

这意味着项目已经从“大范围兼容性摸排阶段”进入“针对性修复求解链运行时行为”的阶段。
