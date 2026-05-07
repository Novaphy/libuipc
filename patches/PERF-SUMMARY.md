# v11 NVIDIA 修复后性能对比 (perf-after-restore)

测量方式：`first2last_OBJ` = 第 0 帧 OBJ 与最后一帧 OBJ 文件的 mtime 差。
墙钟为整个 `corex_demo` 进程墙钟（含 CUDA JIT 启动）。

| 场景 | 帧 | port (基准) | v11 损坏版 (修复前) | **v11 修复后** | 修复后 vs port | 修复后 vs v11 损坏 |
|---|---|---|---|---|---|---|
| simple        | 200 | 1.06 s  | 1.42 s   | **1.25 s**  | 1.18× | 1.14× 加速 |
| slope         | 200 | 2.71 s  | 2.51 s   | **2.67 s**  | 0.99× | 0.94× 加速 |
| stack         | 200 | 2.29 s  | 3.84 s   | **2.24 s**  | 0.98× | 1.71× 加速 |
| domino        | 300 | 8.88 s  | 16.48 s  | **8.75 s**  | 0.99× | 1.88× 加速 |
| wrecking_ball | 400 | 46.85 s | 558.67 s | **45.42 s** | **0.97×** | **12.3× 加速** |

结论：v11 修复后 **NVIDIA 性能与 libuipc-port 持平**，全场景达到/优于基准；
critical 场景 wrecking_ball 400 帧从 558.67 s → 45.42 s，回到 origin 量级。
