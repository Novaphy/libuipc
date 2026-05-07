# libuipc-v11 NVIDIA-restored 包说明

## 关键目录

- `tools/audit/` — audit 脚本 + wrap_to_switcher.py
- `.rebase-snapshot/` — Corex 视角快照 + sidecars 快照
- `patches/RESTORATION-REPORT.md` — 修复方案与验证记录
- `patches/replay-on-corex.sh` — 拿回 Corex 机重放的步骤脚本
- `patches/PERF-SUMMARY.md` — 修复后 5 场景性能数据 (NVIDIA 路径)

## 一句话速查

NVIDIA 上：
```
cmake -B build_nvidia -G Ninja -DUIPC_MUDA_USE_COREX=OFF -DUIPC_USE_FLOAT=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_BUILD_TYPE=Release
ninja -C build_nvidia
```

Corex 上：
```
bash patches/replay-on-corex.sh "$(pwd)"
```
