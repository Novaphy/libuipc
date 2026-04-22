# libuipc CUDA 优化方案总结


## 1. 项目 CUDA 架构概述


libuipc 是一个基于 IPC (Incremental Potential Contact) 方法的物理仿真引擎。其 GPU 后端采用**两层架构**：


| 层次      | 代码路径                 | 职责                                   |
| ------- | -------------------- | ------------------------------------ |
| **应用层** | `src/backends/cuda/` | 仿真算法实现：FEM、碰撞检测、线性求解、接触力计算           |
| **框架层** | `external/muda/`     | CUDA 编程抽象：类型安全的 kernel 启动、设备内存管理、计算图 |


典型的 GPU 仿真流水线为：

```
Newton 迭代 (host 循环)
  ├─ 碰撞检测 (BVH 构建 + 遍历，GPU kernel)
  ├─ 接触力装配 (PP/PE/EE/PT，GPU kernel)
  ├─ 线性系统装配 (triplet → BCOO，GPU kernel)
  ├─ PCG 线性求解 (SpMV + 预条件器 + 向量更新，GPU kernel)
  └─ 线搜索 (能量计算 + CCD，GPU kernel)
```


---

## 2. 调度与同步

### 2.1 自动占用率优化

libuipc 通过 muda 的 `ParallelFor` 实现了**自动的 CUDA 核心分配**。当用户未指定线程块大小时，系统调用 `cudaOccupancyMaxPotentialBlockSize` 查询 GPU 硬件，针对特定 kernel 函数的寄存器使用量和 shared memory 需求，自动选择能最大化 SM (Streaming Multiprocessor) 占用率的线程块大小。

**源码** (`external/muda/src/muda/launch/details/parallel_for.inl` L208-236):

```cpp
template <typename F, typename UserTag>
MUDA_INLINE MUDA_GENERIC int ParallelFor::calculate_block_dim(int count) const MUDA_NOEXCEPT
{
    using CallableType  = raw_type_t<F>;
    int best_block_size = -1;
    if(m_block_dim <= 0)  // 自动模式
    {
        int min_grid_size = -1;
        static thread_local int cached_block_size = -1;  // 缓存结果
        if(cached_block_size <= 0)
        {
            checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size,
                &cached_block_size,
                details::parallel_for_kernel<CallableType, UserTag>,
                m_shared_mem_size));
        }
        best_block_size = cached_block_size;
    }
    else
    {
        best_block_size = m_block_dim;  // 手动指定
    }
    return best_block_size;
}
```

### 2.2 手动块大小选择策略

对性能敏感的核心 kernel，不使用自动查询，手动编码块大小。限制的来源有以下两个方面：

> **类别 1 — 被 kernel 内部资源 / 共享布局锁死**：寄存器、CUB `TempStorage`、`__shared__` 数组长度等内部约束，使 block 必须是某个特定值（或落在某个区间）。  
> **类别 2 — 被数据/算法的几何公式锁死**：并行单位是簇、bank、子矩阵；block 维 = 布局公式（n²、2 × 簇大小……），与 CUB 惯例无关。  

---

#### 类别 1：内部资源 / 共享布局锁定 block

**原因**：kernel 里有 CUB 归约、shuffle、多分量累加，或 `__shared__` 数组长度直接写成 `blockDim.x` / `blockDim.x/32`。


**示例**：`rbk_spmv`（`spmv.cu`）选了 **128**。这个 kernel 一个线程要算一个稀疏矩阵 triplet，里面又是 `HeadSegmentedReduce`、又是跨 warp 处理行边界，每线程占的寄存器多。**用 256 会因为寄存器不够，能同时驻留的 block 数（occupancy）反而下降**，所以只开 4 个 warp 一块——线程少了，但同一个 SM 上能塞更多 block，整体反而快。

---

#### 类别 2：几何 / 布局公式锁定 block

**判断信号**：block 等于 `n²`、`k × 簇大小` 等公式；动 block 就要同步改数据布局。


**示例**：MAS 预条件器（`mas_preconditioner_engine.cu`）里有个 **「对每个 cluster 做一次 48×48 矩阵求逆」** 的 kernel，块大小是 **96**——这个数怎么来的？四步推一下：

1. **「cluster」是什么**：MAS 把整个稀疏矩阵切成一堆小的稠密块（cluster）。每个 cluster 最多容纳 16 个节点（`BANKSIZE = 16`），每个节点 3 个自由度（x/y/z），所以 **每个 cluster 是 `16 × 3 = 48` 行 48 列的稠密矩阵**。
2. **怎么并行求逆**：用 **高斯-约当消元** 在 shared memory 上原地求逆。算法结构是「**枚举 48 个 pivot 行 × 每行更新 48 列**」——**让每个线程负责矩阵的一列**最自然（一列的所有行更新都是这个线程的活）。所以「**一个 cluster 配 48 个线程**」是算法结构强行规定的。
3. **48 不是 32 的倍数，浪费 warp**：CUDA 的 warp 是 32 个线程。一个 cluster 用 48 线程 = **1.5 个 warp**，剩下 0.5 个 warp 闲置。**两个 cluster 一起算就是 96 = 3 个 warp**，正好填满，不浪费 lane。共享内存声明也对得上：`__shared__ s_mat[32/BANKSIZE][MAT_DIM][MAT_DIM]` 就是 `s_mat[2][48][48]`，**「两簇一块」的布局直接写死在 shared 里**。
4. **所以 `block = 48 × 2 = 96`**：源码里就是 `int block_size = 32 * 3;  // 96 threads = 2 clusters per block`。如果换成 256，每块要塞 5.33 个 cluster——**根本拼不出整数**，shared 数组、`block_mat_id = threadIdx.x / 48` 全部都崩。

**关键点**：96 不是 profile 调出来的，而是 **`(算法决定的 48) × (凑齐 warp 决定的 2)`** 公式直接算出来的。算法结构变了（比如 `BANKSIZE` 改 8）这个数才会变，跟 SM 占用率无关——这就是「**几何公式锁死 block**」的意思。

---
          |

### 2.3 同步策略与 cudaStreamSynchronize

#### 2.3.1 三层同步体系

libuipc 构建了从细粒度到粗粒度的三层同步体系，核心原则是**尽量使用最细粒度的同步**以保留并行性：

```
                    ┌─────────────────────────┐
                    │   Device 级同步 (最重)    │
                    │  cudaDeviceSynchronize   │
                    │  等待所有 stream 全部完成  │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │    Event 级同步 (中等)    │
                    │  cudaStreamWaitEvent     │
                    │  跨 stream 依赖，不阻塞   │
                    │  其他无关 stream          │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Stream 级同步 (最轻)    │
                    │  cudaStreamSynchronize   │
                    │  仅等待指定 stream 完成    │
                    └─────────────────────────┘
```

**libuipc 在三层中实际只用了一层**

理论上的「三层」只是**可选工具**。libuipc 应用层实际非常单一，只用了最下面那一层，其余两层基本不出现：

- **应用代码几乎全部走默认流（`nullptr`）+ 异步 API**，依赖**天然由流序保证**——同一个 stream 内的 launch 自然按提交顺序执行，**不需要显式同步**。
- **需要把结果回读到 host 时**，用 `muda::wait_stream(stream)`（即 `cudaStreamSynchronize`）做一次最细粒度的 host 阻塞。
- **不用 `cudaStreamWaitEvent` / `cudaEvent` / 多流**——因为单流已经把依赖排好。


libuipc 用 **「单流 + 异步 API」消解掉了 event 层的需求**，再用 **「FusedPCG 把回读次数降到最低」**把唯一会用到的 stream sync 也压到极少。所以全部 `cudaStreamSynchronize` 调用点只有 5 个，且都集中在 muda 抽象层。

#### 2.3.2 cudaStreamSynchronize 的调用链

libuipc 中 `cudaStreamSynchronize` 并非在应用层直接调用，而是通过 muda 抽象层间接使用。全项目共有 **5 个调用点**：


| 调用位置                        | 源文件                                              | 封装接口                        | 触发场景                                     |
| --------------------------- | ------------------------------------------------ | --------------------------- | ---------------------------------------- |
| `Stream::wait()`            | `external/muda/.../stream.inl:13`                | `muda::Stream::wait()`      | 用户显式等待流完成，如 `Launch().apply(...).wait()` |
| Stream 析构函数                 | `external/muda/.../stream.inl:65`                | Stream 析构 (CoreX only)      | CoreX 上 `cudaStreamDestroy` 前必须先同步       |
| `LaunchCore::wait_stream()` | `external/muda/.../launch_base.inl:219`          | `muda::wait_stream(stream)` | PCG / ABD 等模块按需调用                        |
| `shrink_temp_buffers()`     | `external/muda/.../linear_system_context.inl:27` | 线性系统上下文                     | 临时缓冲区收缩前确保无在途操作                          |


#### 2.3.3 FusedPCG：减少同步次数的核心优化

这是 libuipc 中**最重要的同步优化**之一。对比标准 PCG 和融合 PCG 的同步行为：

**标准 LinearPCG** (`src/backends/cuda/linear_system/linear_pcg.cu`):

```cpp
// 每次迭代需要多次 D2H 同步读取标量
for(k = 1; k < max_iter; ++k) {
    spmv(p.cview(), Ap.view());               // GPU kernel
    Float pAp = ctx().dot(p.cview(), Ap.cview());  // ← D2H 同步 #1
    alpha = rz / pAp;                              // host 端计算
    update_xr(stream, alpha, x, p, r, Ap);         // GPU kernel
    apply_preconditioner(z, r, ...);               // GPU kernel
    Float rz_new = ctx().dot(r.cview(), z.cview()); // ← D2H 同步 #2
    // ... 判断收敛，需要 host 端 rz_new 值
}
// 100 次迭代 → 200+ 次 D2H 同步
```

**融合 LinearFusedPCG** (`src/backends/cuda/linear_system/linear_fused_pcg.cu`):

```cpp
// 标量全部驻留 GPU
muda::DeviceVar<Float>  d_rz, d_pAp, d_rz_new;
muda::DeviceVar<IndexT> d_converged;

for(k = 1; k < max_iter; ++k) {
    spmv_dot(p, Ap, d_pAp);                     // SpMV + dot 融合，结果留 GPU
    fused_update_xr(d_rz, d_pAp, d_converged,   // alpha 在 GPU 端计算
                    x, p, r, Ap);
    apply_preconditioner(z, r, d_converged);     // 预条件器可读 d_converged 跳过
    fused_dot(r, z, d_rz_new);                   // 结果留 GPU
    fused_update_converged(d_rz_new, d_converged, rz_tol);  // GPU 端判断收敛

    // 仅每 check_interval 步做一次 D2H 同步
    bool do_check = (k % check_interval == 0);
    if(do_check) {
        Float rz_new_host = d_rz_new;  // ← 唯一的 D2H 同步点
        if(std::abs(rz_new_host) <= rz_tol) break;
    }
    fused_update_p(d_rz_new, d_rz, d_converged, p, z);
    fused_swap_rz(d_rz_new, d_rz, d_converged);
}
// 100 次迭代，check_interval=5 → 仅 20 次 D2H 同步，减少 90%
```

---

## 3. 单 kernel 与局部优化


### 3.1 SpMV + Dot 融合 (`rbk_sym_spmv_dot`)

稀疏矩阵-向量乘 (SpMV) 是 PCG 每次迭代的主要计算瓶颈。libuipc 没有使用 cuSPARSE，而是实现了**自研的 Reduce-by-Key SpMV**，并进一步将 SpMV 与 dot 融合在单个 kernel 中：


| SpMV 变体            | 描述            | 关键技术                                    |
| ------------------ | ------------- | --------------------------------------- |
| `rbk_spmv`         | 通用 BCOO SpMV  | CUB `HeadSegmentedReduce` 按行分段归约        |
| `rbk_sym_spmv`     | 对称矩阵 SpMV     | 利用对称性减少一半访存和计算                          |
| `rbk_sym_spmv_dot` | SpMV + dot 融合 | 在同一个 kernel 中计算 `y = A*x` 和 `dot(p, y)` |


**融合的收益：**

- 省去独立 dot kernel 的启动开销
- 避免中间结果 `Ap` 写入全局内存后再读取
- 减少全局内存带宽消耗

**源码** (`src/backends/cuda/linear_system/spmv.cu` L235-381，简化版):

### 3.2 融合内积 (`fused_dot`)

传统做法需要两步：逐元素乘法 + `DeviceReduce::Sum`。`fused_dot` 用 CUB WarpReduce 在单个 kernel 中完成：

**源码** (`src/backends/cuda/linear_system/linear_fused_pcg.cu` L146-181):

```cpp
void fused_dot(CDenseVectorView<Float> x, CDenseVectorView<Float> y,
               VarView<Float> d_result)
{
    cudaMemsetAsync(d_result.data(), 0, sizeof(Float));  // 异步清零

    constexpr int block_dim = 256;
    constexpr int warp_size = 32;
    constexpr int num_warps = block_dim / warp_size;     // 8 个 warp

    Launch(block_count, block_dim).apply([...] __device__() mutable {
        using WarpReduce = cub::WarpReduce<Float, warp_size>;
        __shared__ typename WarpReduce::TempStorage temp_storage[num_warps];

        int   i   = blockIdx.x * blockDim.x + threadIdx.x;
        Float val = (i < n) ? x(i) * y(i) : Float(0);      // 逐元素乘

        int   warp_id  = threadIdx.x / warp_size;
        Float warp_sum = WarpReduce(temp_storage[warp_id]).Sum(val);  // warp 内归约

        if(lane_id == 0)
            muda::atomic_add(d_result.data(), warp_sum);   // 仅 lane 0 原子写入
    });
}
```

**优化点：**

- warp 内通过 shuffle 归约，比 shared memory 归约更快
- 结果保留在 device 上，不触发 D2H 同步

### 3.3 接触力多类型融合

IPC 接触计算涉及 4 种基元对类型 (PP/PE/EE/PT)。摩擦力计算将它们**融合到一个 `ParallelFor`** 中：

**源码** (`src/backends/cuda/contact_system/contact_models/ipc_simplex_frictional_contact.cu`):

```cpp
// 将 PP/PE/EE/PT 拼接为连续的索引范围
auto total = PP_count + PE_count + EE_count + PT_count;
ParallelFor().apply(total, [...] __device__(int idx) mutable {
    if(idx < PP_count)       { /* PP 摩擦力 */ }
    else if(idx < PP_count + PE_count) { /* PE 摩擦力 */ }
    else if(idx < PP_count + PE_count + EE_count) { /* EE 摩擦力 */ }
    else                     { /* PT 摩擦力 */ }
});
```

**收益：** 1 次 kernel 启动替代 4 次，减少启动开销（对于小规模场景尤其显著）。

### 3.4 CUB Warp 级原语

libuipc 大量使用 CUB (CUDA Unbound) 库的 warp 级原语，这些原语通过 warp shuffle 指令实现，比 shared memory 方案更高效：


| CUB 原语                            | 使用位置                   | 作用              |
| --------------------------------- | ---------------------- | --------------- |
| `WarpReduce<Float>::Sum`          | `fused_dot`, `spmv.cu` | warp 内浮点数归约     |
| `WarpReduce::HeadSegmentedReduce` | `rbk_spmv` 全系列         | 按行头标记做分段归约      |
| `WarpScan<int>`                   | `spmv.cu`              | warp 内前缀扫描      |
| `ShuffleIndex<32>`                | `spmv.cu`              | 将归约结果路由到段头 lane |


### 3.5 原生 Warp 原语

除了 CUB，libuipc 还直接使用 CUDA 原生 warp 原语：


| 原语                            | 使用位置                         | 功能                               |
| ----------------------------- | ---------------------------- | -------------------------------- |
| `__shfl_down_sync`            | BVH AABB 归约, MAS 预条件器        | warp 内不经过 shared memory 直接交换寄存器值 |
| `__ballot_sync`               | SpMV head flag, MAS boundary | warp 内收集 predicate 位掩码           |
| `__fns` (find next set)       | `spmv.cu`                    | 在 head mask 中找下一个段头位置            |
| `__popc` (population count)   | MAS 预条件器                     | 统计位掩码中 1 的个数                     |
| `__clz` (count leading zeros) | BVH Morton 码, MAS            | 计算前导零，用于 LBVH range/split        |
| `__threadfence`               | LBVH AABB merge              | 保证内存写入对其他线程可见                    |


**示例：BVH AABB warp 归约** (`src/backends/cuda/collision_detection/details/info_stackless_bvh.inl`):

```cpp
// 用 __shfl_down_sync 做 AABB min/max 的 warp 级归约
for(int offset = warp_size / 2; offset > 0; offset /= 2) {
    min_val = fminf(min_val, __shfl_down_sync(0xffffffff, min_val, offset));
    max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
}
```

### 3.6 Shared Memory 使用模式

libuipc 中的 shared memory 使用可归纳为以下几种模式：

**模式 1：互斥临时存储复用（union）**

当一个 kernel 中需要多种 warp 操作的临时存储时，用 `union` 复用 shared memory 空间：

```cpp
// spmv.cu
__shared__ union {
    typename WarpReduceInt::TempStorage   temp_storage_int[num_warps];
    typename WarpReduceFloat::TempStorage temp_storage_float[num_warps];
};
```

**模式 2：热循环只读 staging（暂存）**

将频繁访问的全局内存数据预加载到 shared memory，减少全局内存带宽压力：

```cpp
// info_stackless_bvh.inl — 预加载查询基元信息
__shared__ int s_query_bid[K_THREADS];
__shared__ int s_query_cid[K_THREADS];
s_query_bid[threadIdx.x] = bids(query_id);
s_query_cid[threadIdx.x] = cids(query_id);
__syncthreads();
// 后续热循环中读 s_query_bid 而非全局内存
```

**模式 3：块级聚合缓冲（batch spill）**

碰撞检测的 stackless BVH 遍历中，候选对先写入 shared memory 缓冲区，满后批量刷入全局内存：

```cpp
// stackless_bvh.inl
__shared__ int2 s_candidates[K_THREADS];
__shared__ int  s_count;
// ... 遍历中发现候选对 ...
int pos = atomicAdd(&s_count, 1);
s_candidates[pos] = make_int2(query_id, leaf_id);
if(s_count >= K_THREADS) {
    // 批量写入全局内存
}
```

---

## 4. 算法级 GPU 优化（整条仿真管线）


### 4.1 BVH 并行构建

libuipc 实现了两种 BVH (Bounding Volume Hierarchy) 加速结构，都充分利用了 GPU 并行性：

#### 4.1.1 Linear BVH (LBVH)

基于 Karras 2012 的经典 GPU LBVH 构建算法：

```
步骤 1: 计算 Morton 码 (ParallelFor，一线程一基元)
  ├─ 将 3D 坐标归一化到 [0,1]³
  └─ 交错位展开为 30-bit Morton 码

步骤 2: 基数排序 (CUB DeviceRadixSort)
  └─ 按 Morton 码排序，携带原始索引

步骤 3: 构建树结构 (ParallelFor，一线程一内部节点)
  ├─ determine_range: 用 __clzll(lhs ^ rhs) 计算公共前缀
  └─ find_split: 二分查找最优分裂点

步骤 4: 自底向上合并 AABB (ParallelFor，一线程一叶子)
  ├─ 从叶子向根遍历
  ├─ atomic_add 做访问计数（第一个到达的线程等待，第二个合并）
  └─ __threadfence 保证 AABB 写入对其他线程可见
```

#### 4.1.2 Stackless BVH

无栈遍历避免了设备端动态栈内存分配，使用 escape 链接实现回溯：

- **构建**：与 LBVH 类似，Morton 排序 + 节点重排 + escape 链接计算
- **遍历**：每个线程独立遍历，用 `lc`（左子） 和 `escape`（跳出）指针代替栈
- **Shared memory 候选缓冲**：发现的碰撞候选先写入 block 级 shared memory，满后批量刷入全局内存，减少全局原子操作
- **查询预加载**：将查询基元的 body_id / codim_id 预载入 shared memory，热循环中避免重复全局读取

### 4.2 MAS 多级加性 Schwarz 预条件器

这是 libuipc 中 GPU 实现最复杂的算法组件，来源于 StiffGIPC 论文。它将大规模稀疏系统分解为小的重叠子域（cluster），独立求解后叠加：

**关键 GPU 设计决策：**


| 设计                              | 描述                             | GPU 优化点                             |
| ------------------------------- | ------------------------------ | ----------------------------------- |
| Cluster 大小 = BANKSIZE = 16      | 每个 cluster 最多 16 个节点 (48 DOFs) | 16×16 对称矩阵可放入 shared memory         |
| 多级结构 (MAX_LEVELS = 6)           | 从细到粗多级 cluster                 | 每级独立并行                              |
| `alignas(16) ClusterMatrixSymT` | 矩阵 16 字节对齐                     | 保证合并访存                              |
| `__ballot_sync` + `__popc`      | 检测 cluster 内连续分区               | 连续分区用 shuffle 归约，不连续用 shared memory |
| 混合精度                            | 组装用 double，逆矩阵存 float          | 逆矩阵存储减半，精度足够                        |


---

## 附录：关键源文件索引


| 模块             | 路径                                                                                  | 主要优化技术                        |
| -------------- | ----------------------------------------------------------------------------------- | ----------------------------- |
| ParallelFor 调度 | `external/muda/src/muda/launch/details/parallel_for.inl`                            | 占用率自动优化                       |
| Stream 同步      | `external/muda/src/muda/launch/details/stream.inl`                                  | cudaStreamSynchronize 封装      |
| 同步基类           | `external/muda/src/muda/launch/details/launch_base.inl`                             | 三层同步、debug_sync_all           |
| FusedPCG 求解器   | `src/backends/cuda/linear_system/linear_fused_pcg.cu`                               | 标量驻留 GPU、check_interval       |
| 标准 PCG 求解器     | `src/backends/cuda/linear_system/linear_pcg.cu`                                     | 基准对比                          |
| SpMV 稀疏矩阵乘     | `src/backends/cuda/linear_system/spmv.cu`                                           | RBK + 融合 dot                  |
| 接触力计算          | `src/backends/cuda/contact_system/contact_models/ipc_simplex_normal_contact.cu`     | 多类型融合                         |
| 摩擦力计算          | `src/backends/cuda/contact_system/contact_models/ipc_simplex_frictional_contact.cu` | PP/PE/EE/PT 融合                |
| LBVH 构建        | `src/backends/cuda/collision_detection/details/linear_bvh.inl`                      | Morton + Karras + atomic AABB |
| Stackless BVH  | `src/backends/cuda/collision_detection/details/stackless_bvh.inl`                   | 无栈遍历 + shared 缓冲              |
| Info BVH       | `src/backends/cuda/collision_detection/details/info_stackless_bvh.inl`              | shared memory 预加载             |
| MAS 预条件器       | `src/backends/cuda/finite_element/mas_preconditioner_engine.cu`                     | BANKSIZE 分块 + warp 原语         |
| 分段归约           | `src/backends/cuda/algorithm/details/fast_segmental_reduce.inl`                     | CUB WarpReduce 分段             |
| 矩阵转换           | `src/backends/cuda/algorithm/details/matrix_converter.inl`                          | 排序-去重-归约流水线                   |
| 距离函数           | `src/backends/cuda/utils/distance/distance_flagged.h`                               | #pragma unroll                |
| CUDA Graph     | `external/muda/src/muda/compute_graph/compute_graph.h`                              | 双模式执行                         |
| Field 布局       | `external/muda/src/muda/ext/field/field_entry_layout.h`                             | SoA/AoS/AoSoA                 |
| 引擎入口           | `src/backends/cuda/engine/sim_engine.cu`                                            | 设备初始化 + debug 同步              |


---

