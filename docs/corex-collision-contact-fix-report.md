# libuipc CoreX/天数 GPU 适配 — 碰撞检测与接触力修复工作总结

## 一、项目背景

本轮工作衔接上一轮的基础修复（仿真静止、四面体瞬移、Newton 不收敛），在已经恢复基本自由落体仿真的基础上，**开启碰撞检测（Contact）和接触力组装功能**，并逐步修复 CoreX 环境下的新故障。

上一轮工作结论：在 contact 关闭的 simple 场景下，200 帧自由落体正常运行，Newton 收敛正常。

本轮目标：开启 contact（先不开 friction），使两个四面体之间能正常发生碰撞、接触力生效，仿真物理行为正确。

---

## 二、场景配置调整

### 2.1 开启碰撞检测

`apps/examples/corex_demo/main.cpp` 中 simple 场景配置：

```cpp
config["contact"]["enable"]             = 1;
config["contact"]["friction"]["enable"] = 0;
```

### 2.2 缩小初始间距

将上方四面体的初始高度偏移从 `Vector3::UnitY() * 1.3` 改为 `Vector3::UnitY() * 1.005`，使两个四面体在第一帧就有接触对（PP/PE/EE/PT），便于快速验证碰撞检测和接触力路径。

---

## 三、问题一：Stackless BVH 碰撞检测 ParallelFor 静默失败

### 现象

开启 contact 后，仿真第一帧检测到 0 个碰撞对（PTs=0, EEs=0, PEs=0, PPs=0），实际两个四面体已有重叠。

### 根因

`stackless_bvh.inl` 中的 BVH 构建函数（Morton code 计算、逆映射、图元构建、分裂度量、内部节点排序、BVH 节点重排序等）全部使用 `muda::ParallelFor` 的设备端 lambda，在 CoreX 上静默失败，导致 BVH 树完全无效。

此外 `muda::BufferLaunch().fill()` 在计数器清零时也静默失败。

### 解决方式

在 `stackless_bvh.inl` 中：

1. 添加 `namespace corex_bvh`，编写 7 个 `static __global__` kernel 替代 `ParallelFor` lambda：

| Kernel | 功能 |
|--------|------|
| `kernel_calcMCs` | 计算 Morton code |
| `kernel_calcInverseMapping` | 计算逆映射表 |
| `kernel_buildPrimitives` | 从 AABB 构建图元 |
| `kernel_calcSplitMetrics` | 计算分裂度量 |
| `kernel_calcIntNodeOrders` | 计算内部节点排序 |
| `kernel_updateBvhExtNodeLinks` | 更新外部节点链接 |
| `kernel_reorderNode` | 重排序 BVH 节点 |

2. 将 `BufferLaunch().fill()` 碰撞计数器清零替换为 `cudaMemset()`。

3. 添加 `RAW_PTR(x)` 宏简化 `thrust::raw_pointer_cast` 调用。

### 关键发现

BVH 的 **查询阶段**（`detect` / `query`）使用 `muda::Launch().apply()` 的设备 lambda 在 CoreX 上**可以正常工作**。这是因为 `Launch().apply()` 与 `ParallelFor().apply()` 在底层的设备代码注册方式不同。后续修复中，**`Launch` 成为替代 `ParallelFor` 的首选方案**（优于编写独立 `__global__` kernel）。

---

## 四、问题二：轨迹过滤器 AABB 构建 ParallelFor 失败

### 现象

BVH 修复后碰撞检测仍然报 0 对，因为轨迹过滤器的 AABB 计算结果全为零。

### 根因

`stackless_bvh_simplex_trajectory_filter.cu` 中为点、边、三角形构建运动扫掠 AABB 的 `ParallelFor` 全部静默失败。

### 解决方式

在 `namespace corex_filter` 中编写 3 个 `__global__` kernel：

| Kernel | 功能 |
|--------|------|
| `kernel_build_point_aabbs` | 点的轨迹扩展 AABB |
| `kernel_build_edge_aabbs` | 边的轨迹扩展 AABB |
| `kernel_build_triangle_aabbs` | 三角形的轨迹扩展 AABB |

修复后，第一帧成功检测到接触对：`PTs: 0, EEs: 17, PEs: 25, PPs: 25`。

---

## 五、问题三：IPC 接触力梯度/Hessian 组装 — `muda::Launch` 复杂 lambda 静默失败

### 现象

碰撞检测正常后，仿真在 `IPCSimplexNormalContact::do_assemble()` 阶段挂死。`cudaDeviceSynchronize()` 无限阻塞。

### 排查过程

#### 第 1 步：定位挂死位置

通过添加 `fmt::println(stderr, ...)` + `std::fflush(stderr)` + `cudaDeviceSynchronize()` 诊断，定位到挂死发生在 `do_assemble()` 中的 `muda::ParallelFor().apply()` 调用。

#### 第 2 步：替换为 `muda::Launch().apply()`

将 `ParallelFor` 替换为 `Launch(grid_dim, block_dim).apply()`，手动计算线程索引。编译通过但仍然挂死。

#### 第 3 步：拆分为独立 kernel

将原来的融合 kernel（一次处理 PP+PE+EE+PT 四种接触类型）拆分为 4 个独立的 `Launch` 调用，每种接触类型一个，减少单个 kernel 的复杂度。

逐个 kernel 添加诊断后发现：
- **空 lambda**（noop kernel）：正常执行 ✓
- **读取 viewer 数据**（如读取 PPs、positions）：正常执行 ✓
- **完整物理计算**（barrier gradient/Hessian、`make_spd`）：挂死 ✗

#### 第 4 步：验证 assembler 写入路径

将 4 个 kernel 中的物理计算全部替换为写入零矩阵（`Eigen::Matrix::Zero()`），仅保留 `DoubletVectorAssembler` 和 `TripletMatrixAssembler` 的写入调用：

```cpp
// PP kernel — 写零梯度/Hessian 替代实际计算
const auto& PP = PPs(i);
Vector6 G = Vector6::Zero();
DoubletVectorAssembler DVA{PP_Gs};
DVA.segment<2>(i * 2).write(PP, G);
if(!gradient_only)
{
    Matrix6x6 H = Matrix6x6::Zero();
    TripletMatrixAssembler TMA{PP_Hs};
    TMA.half_block<2>(i * PPHalfHessianSize).write(PP, H);
}
```

PE、EE、PT kernel 同理（分别用 `Vector9/Matrix9x9`、`Vector12/Matrix12x12`）。

### 结果

4 个 kernel 全部成功执行并同步：

```
[ipc_assemble] PP launch (n=25) ... PP sync OK
[ipc_assemble] PE launch (n=25) ... PE sync OK
[ipc_assemble] EE launch (n=17) ... EE sync OK
[ipc_assemble] all done
```

### 根因分析

CoreX 上 `muda::Launch().apply()` 可以正常执行**简单**的设备 lambda（数据读写、viewer 访问），但**无法执行包含复杂计算的 lambda**（如 IPC barrier gradient/Hessian 计算、`make_spd` 等）。推测原因为：

1. **寄存器压力过大**：IPC 计算涉及大量 Eigen 矩阵运算（`Matrix12x12`、`Matrix6x6` 等），CoreX 的寄存器文件可能不足以支撑这种复杂度的 lambda
2. **编译器代码生成问题**：CoreX 的 clang CUDA 编译器对复杂 device lambda 的代码生成可能存在缺陷

### 当前状态（待解决）

接触力的**能量计算**（`do_compute_energy`）仍使用完整物理公式，通过 `Launch().apply()` 执行。梯度和 Hessian 组装当前**写入全零**，接触力实际未生效。

需要后续将完整的 barrier gradient/Hessian 计算拆分为独立的 `__global__` kernel 函数，使用裸指针参数传递数据，完全避开 device lambda 捕获。

---

## 六、问题四：矩阵转换器 `MatrixConverter::convert()` — 多处 ParallelFor 和 FastSegmentalReduce 失败

### 现象

接触力组装通过（写零）后，仿真在 `MatrixConverter::convert()` 中挂死，将 Triplet 格式转换为 BCOO/BSR 格式时出错。

### 排查过程

#### 阶段 1：定位初始挂死

添加诊断后发现 `convert()` 入口处的 `cudaDeviceSynchronize()` 就挂死。原因是上游（contact assembler）的 kernel 仍在挂死状态。解决 contact 问题（写零）后，此处不再挂死。

#### 阶段 2：`static __global__` 在新 `.cu` 文件中失败

最初将 `matrix_converter.inl` 中的 `ParallelFor` 替换为在 `matrix_converter.inl` 中定义的 `static __global__` kernel，但因该 `.inl` 被多个翻译单元包含，导致链接器问题。

改为在新文件 `corex_matrix_converter_kernels.cu` 中定义 kernel，但**即使是最简单的 noop kernel 也挂死**。CoreX 的 CUDA 运行时无法正确注册新 `.cu` 文件中动态加载共享库的设备代码。

#### 阶段 3：移动 kernel 到已有 `.cu` 文件

将所有矩阵转换器 kernel 移到 `global_dytopo_effect_manager.cu`（一个已有的、编译正常的 `.cu` 文件）中。`corex_matrix_converter_kernels.cu` 保留为空（仅有注释说明原因）。

### 解决方式

在 `algorithm/details/matrix_converter.inl` 中，以下操作全部替换：

**Triplet → BCOO 转换**（`_radix_sort_indices_and_blocks`, `_make_unique_indices`, `_make_unique_block_warp_reduction`）：

| 原始操作 | CoreX 替换 |
|----------|-----------|
| `ParallelFor`: hash (row,col) → uint64 | `corex_matconv::launch_hash_ij` |
| `ParallelFor`: decode hash → ij_pairs | `corex_matconv::launch_decode_hash` |
| `ParallelFor`: copy sorted blocks | `corex_matconv::launch_copy_sorted_blocks_3x3` |
| `ParallelFor`: write unique ij | `corex_matconv::launch_write_unique_ij` |
| `BufferLaunch().fill()` + `ParallelFor`: mark partition | `cudaMemset` + `corex_matconv::launch_mark_partition` |
| `FastSegmentalReduce<>`: 合并重复 3×3 块 | `corex_matconv::launch_segmental_reduce_3x3` |

**BCOO 就地排序**（`_radix_sort_indices_and_blocks` overload）：

| 原始操作 | CoreX 替换 |
|----------|-----------|
| `ParallelFor`: hash ij | `corex_matconv::launch_hash_ij` |
| `ParallelFor`: decode hash | `corex_matconv::launch_decode_hash` |
| `ParallelFor`: copy blocks + ij | `corex_matconv::launch_copy_sorted_blocks_with_ij_3x3` |

**BCOO → BSR 转换**（`_calculate_block_offsets`）：

| 原始操作 | CoreX 替换 |
|----------|-----------|
| `ParallelFor`: scatter col counts | `corex_matconv::launch_scatter_col_counts` |

**DoubletVector → BCOOVector 转换**（`_make_unique_indices`, `_make_unique_segment_warp_reduction`）：

| 原始操作 | CoreX 替换 |
|----------|-----------|
| `ParallelFor`: write unique indices | `corex_matconv::launch_write_unique_indices` |
| `BufferLaunch().fill()` + `ParallelFor`: mark partition | `cudaMemset` + `corex_matconv::launch_mark_partition` |
| `FastSegmentalReduce<64,32>`: 合并向量段 | `corex_matconv::launch_segmental_reduce_3x1` |

### FastSegmentalReduce 的特殊处理

`FastSegmentalReduce` 使用 `muda::Launch().apply()` 内含 `cub::WarpReduce` + Eigen 矩阵运算 + 共享内存的复杂 lambda，在 CoreX 上挂死。

替换为使用 `atomicAdd` 的简单 `__global__` kernel：

```cpp
// 每个线程将自己的 3×3 block 原子加到对应 segment 的输出
static __global__ void kernel_segmental_reduce_3x3(int N, const int* segment_ids,
                                                    const BlockT3* in_blocks,
                                                    BlockT3* out_blocks)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int seg = segment_ids[i];
    const BlockT3& val = in_blocks[i];
    double* dst = out_blocks[seg].data();
    const double* src = val.data();
    for(int j = 0; j < 9; ++j)
        corex_atomic_add_double(&dst[j], src[j]);
}
```

由于 CoreX 不支持 `double` 类型的 `atomicAdd`，使用 CAS 循环实现：

```cpp
__device__ __forceinline__ void corex_atomic_add_double(double* address, double val)
{
    unsigned long long* address_as_ull = reinterpret_cast<unsigned long long*>(address);
    unsigned long long  old_val = *address_as_ull;
    unsigned long long  assumed;
    do {
        assumed = old_val;
        old_val = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while(assumed != old_val);
}
```

### 新文件

| 文件 | 内容 |
|------|------|
| `algorithm/corex_matrix_converter_kernels.h` | 声明 `corex_matconv` 命名空间下的 10 个 `launch_*` 函数和 `BlockT3`/`VecT3` 类型别名 |
| `algorithm/corex_matrix_converter_kernels.cu` | 空文件（注释说明 kernel 实现在 `global_dytopo_effect_manager.cu` 中） |

### `fast_segmental_reduce.inl` 修改

将 `BufferLaunch().fill<T>(out, T{0})` 替换为 `cudaMemsetAsync(...)` 用于标量和矩阵两个重载版本的输出缓冲区清零。

### 当前状态（存疑）

在测试中发现 `launch_segmental_reduce_3x3` 的调试输出未出现，但 `FastSegmentalReduce` 替换代码的 `#if UIPC_COREX_CUDA10_COMPAT` 条件编译确认正确。问题可能是编译缓存未完全刷新或 `.inl` 文件的模板实例化路径差异，需要进一步排查。仿真在 `warp_reduce: done` 后的 `cudaDeviceSynchronize()` 处挂死，即 `FastSegmentalReduce` 的 `Launch().apply()` 提交的 kernel 未能完成。

---

## 七、问题五：CoreX 新 `.cu` 文件设备代码注册失败

### 现象

在动态加载的共享库（`libuipc_backend_cuda.so`）中新增 `.cu` 文件时，即使是最简单的 `__global__ void noop() {}` kernel 也无法执行，`cudaDeviceSynchronize()` 永久挂死。

### 根因

CoreX 的 CUDA 运行时在加载动态共享库时，对设备代码的注册（`__cudaRegisterFunction`）存在缺陷。已有的 `.cu` 文件中的 kernel 可以正常注册和执行，但新增 `.cu` 文件的 kernel 无法被运行时发现。

### 解决方式

将所有需要新增的 `__global__` kernel 放入**已有的** `.cu` 文件中。本轮选择 `global_dytopo_effect_manager.cu` 作为宿主文件，所有矩阵转换器 kernel 的实现均放在此文件末尾。

---

## 八、`ipc_vertex_half_plane_normal_contact.cu` 修改

将 `do_compute_energy()` 和 `do_assemble()` 中的 `ParallelFor().apply()` 替换为 `Launch(grid, block).apply()`，手动进行线程索引计算。由于当前测试场景中 VertexHalfPlane 接触对数量为 0，这些 kernel 实际未执行。

---

## 九、全部修改文件汇总（本轮）

```
apps/examples/corex_demo/main.cpp
    — 开启 contact，缩小初始间距

src/backends/cuda/collision_detection/details/stackless_bvh.inl
    — 7 个 corex_bvh kernel 替代 ParallelFor，cudaMemset 替代 BufferLaunch().fill()

src/backends/cuda/collision_detection/filters/stackless_bvh_simplex_trajectory_filter.cu
    — 3 个 corex_filter kernel 替代 AABB 构建 ParallelFor

src/backends/cuda/contact_system/contact_models/ipc_simplex_normal_contact.cu
    — ParallelFor → Launch，拆分为 PP/PE/EE/PT 四个独立 Launch
    — 梯度/Hessian 暂写零（复杂物理计算 lambda 在 CoreX 上挂死）

src/backends/cuda/contact_system/contact_models/ipc_vertex_half_plane_normal_contact.cu
    — ParallelFor → Launch

src/backends/cuda/algorithm/details/matrix_converter.inl
    — 所有 ParallelFor/BufferLaunch/FastSegmentalReduce 替换为 corex_matconv 函数

src/backends/cuda/algorithm/corex_matrix_converter_kernels.h   （新文件）
    — corex_matconv 函数声明

src/backends/cuda/algorithm/corex_matrix_converter_kernels.cu  （新文件，空）
    — 占位，实现在 global_dytopo_effect_manager.cu

src/backends/cuda/algorithm/details/fast_segmental_reduce.inl
    — BufferLaunch().fill() → cudaMemsetAsync

src/backends/cuda/dytopo_effect_system/global_dytopo_effect_manager.cu
    — 添加 12 个 corex_matconv __global__ kernel
    — corex_atomic_add_double CAS 实现
    — 添加 _assemble / _convert_matrix 诊断日志
```

---

## 十、当前仿真状态

| 阶段 | 状态 |
|------|------|
| BVH 碰撞检测 | ✅ 正常工作（第一帧 PP=25, PE=25, EE=17） |
| 轨迹过滤器 AABB | ✅ 正常工作 |
| 接触力能量计算 (`do_compute_energy`) | ✅ 使用 `Launch().apply()` 执行完整物理计算 |
| 接触力梯度/Hessian (`do_assemble`) | ⚠️ kernel 可执行但当前写零，物理计算 lambda 挂死 |
| 矩阵格式转换 (Triplet→BCOO→BSR) | ⚠️ 大部分环节通过，`FastSegmentalReduce` 替换未生效待排查 |
| 线性求解 (PCG) | ⏸️ 未到达（被上游阻塞） |
| 摩擦力 | ⏸️ 尚未开启 |

---

## 十一、待解决事项

### 高优先级

1. **排查 `FastSegmentalReduce` 替换未生效问题**：`#if UIPC_COREX_CUDA10_COMPAT` 条件在 `_make_unique_block_warp_reduction` 中似乎走了 `#else` 路径（旧的 `FastSegmentalReduce` 仍在执行），导致 `cudaDeviceSynchronize()` 挂死。需要确认模板实例化时宏定义是否有效。

2. **将 IPC 接触力计算改写为 `__global__` kernel**：当前 PP/PE/EE/PT 的梯度和 Hessian 写零，需要将 barrier gradient/Hessian 的完整计算（包括 `PP_barrier_gradient_hessian`、`PE_barrier_gradient_hessian`、`EE_barrier_gradient_hessian`、`PT_barrier_gradient_hessian`、`make_spd`）改写为独立的 `__global__` kernel 函数。

### 中优先级

3. **修复 CFL / 自适应接触参数**（`global_contact_manager.cu`）：其中的 `ParallelFor` 也需要替换。

4. **验证 200 帧碰撞仿真**：接触力生效后，验证两个四面体碰撞、弹开的物理行为是否正确。

### 低优先级

5. **开启 friction**：在 contact 完全正常后开启摩擦力，修复摩擦相关的 `ParallelFor` 调用。

6. **性能优化**：当前所有 CoreX kernel 中的 `cudaDeviceSynchronize()` 可以在功能验证完成后改为异步执行。

---

## 十二、核心经验总结（本轮新增）

1. **`muda::Launch().apply()` 在 CoreX 上有限可用** — 简单 lambda（数据读写、viewer 访问）可以正常工作，但包含复杂计算（大量 Eigen 矩阵运算、CUB WarpReduce、共享内存）的 lambda 会静默挂死。这与上一轮发现的 `ParallelFor` 完全失败不同，`Launch` 是部分可用的。

2. **CoreX 无法注册新 `.cu` 文件的设备代码** — 在动态加载的共享库中，新增的 `.cu` 文件中的 `__global__` kernel 无法被 CUDA 运行时发现。解决方案是将 kernel 实现放入已有的 `.cu` 文件中。

3. **CoreX 不支持 `double` 类型的 `atomicAdd`** — 编译器报 `cannot compile this builtin function yet`。需要使用 `atomicCAS` 实现的 CAS 循环替代。

4. **`FastSegmentalReduce` 的 warp reduction lambda 在 CoreX 上挂死** — 因为内含 `cub::WarpReduce` + Eigen 矩阵操作 + 共享内存的复杂 lambda。替换为简单的 atomic 累加 kernel。

5. **渐进式诊断方法有效** — 先写零验证 kernel 调度和数据写入路径正确，再逐步恢复实际计算逻辑，是定位 CoreX lambda 复杂度限制的有效策略。
