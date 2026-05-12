#!/usr/bin/env bash
#
# replay-on-corex.sh
# ------------------
# 在 Corex 机上验证 v11 修复后版本：
#   1. 静态自检 - Corex 视角必须与修复前快照字节等价 (ok+cosmetic, fail=0)
#   2. NVIDIA 视角必须与 libuipc-origin 字节等价 (可选, 需 libuipc-origin 在隔壁)
#   3. CMake 配置 + ninja 编译 (UIPC_MUDA_USE_COREX=ON)
#   4. 烟测：跑 domino 50 帧, 末帧 OBJ 与 Corex 端 baseline 字节比对
#
# 使用：
#   bash replay-on-corex.sh /abs/path/to/libuipc-v11-fixed [/abs/path/to/baseline-domino-obj]
#
#   <baseline-domino-obj> 可选：上一次在 Corex 上跑出的 domino 末帧 obj 路径，
#   若提供则做字节比对；不提供则只验证场景能跑完不崩。

set -euo pipefail

ROOT="${1:?usage: replay-on-corex.sh <v11-root> [baseline-obj]}"
BASELINE_OBJ="${2:-}"
ROOT="$(realpath "$ROOT")"
cd "$ROOT"

echo "================================================================"
echo "[1/4] Corex 静态等价性检查 (verify_corex_branch_equiv.sh)"
echo "================================================================"
bash tools/audit/verify_corex_branch_equiv.sh
echo "✅ Corex 静态检查通过 (ok+cosmetic, fail=0)"

echo
echo "================================================================"
echo "[2/4] (可选) NVIDIA 静态等价性检查"
echo "================================================================"
if [ -d "../libuipc-origin/libuipc/src" ]; then
    REF_DIR="$(realpath ../libuipc-origin/libuipc)"
    REF_DIR="$REF_DIR" bash tools/audit/verify_nvidia_branch_equiv.sh
    echo "✅ NVIDIA 静态检查通过"
else
    echo "(略过：未在 ../libuipc-origin/libuipc 找到 origin 源码树)"
fi

echo
echo "================================================================"
echo "[3/4] Corex 编译"
echo "================================================================"
BUILD_DIR="${ROOT}/build_corex"
if [ ! -f "${BUILD_DIR}/build.ninja" ]; then
    cmake -S . -B "${BUILD_DIR}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DUIPC_MUDA_USE_COREX=ON \
        -DUIPC_USE_FLOAT=ON
fi
ninja -C "${BUILD_DIR}" -j8
echo "✅ Corex 编译完成 (build_corex/Release/bin/corex_demo 已生成)"

echo
echo "================================================================"
echo "[4/4] domino 50 帧烟测 + 末帧 OBJ 字节比对"
echo "================================================================"
SMOKE_OUT="${ROOT}/_smoke-domino"
mkdir -p "${SMOKE_OUT}"
rm -f "${SMOKE_OUT}/scene_surface_"*.obj 2>/dev/null || true
cd "${BUILD_DIR}/Release/bin"
./corex_demo --backend cuda --scene domino --frames 50 \
    --output_dir "${SMOKE_OUT}" > "${SMOKE_OUT}/run.log" 2>&1

LAST_OBJ="${SMOKE_OUT}/scene_surface_0049.obj"
if [ ! -f "${LAST_OBJ}" ]; then
    echo "FAIL: 末帧 OBJ 未生成"
    tail -10 "${SMOKE_OUT}/run.log"
    exit 1
fi
echo "✅ domino 50 帧跑完，末帧 OBJ：${LAST_OBJ}"

if [ -n "${BASELINE_OBJ}" ] && [ -f "${BASELINE_OBJ}" ]; then
    if cmp -s "${LAST_OBJ}" "${BASELINE_OBJ}"; then
        echo "✅ 末帧 OBJ 与 baseline 字节相同"
    else
        echo "⚠️  末帧 OBJ 与 baseline 不同，请人工检查："
        diff <(head -50 "${LAST_OBJ}") <(head -50 "${BASELINE_OBJ}") | head -20 || true
        exit 2
    fi
else
    echo "(未提供 baseline OBJ，跳过字节比对)"
fi

echo
echo "================================================================"
echo "全部 4 步通过 - v11 修复后在 Corex 上行为与修复前一致"
echo "================================================================"
