#!/usr/bin/env bash
#
# corex_remote_run.sh
# -------------------
# Drive a corex / Iluvatar-side regression run after the 2026-04-22 NVIDIA-rebase
# of libuipc-port. Builds the corex backend, runs the 5 acceptance scenes with
# the same frame counts as the existing nvidia-results/ baseline, and emits a
# diff report against the OBJ baseline (last-frame vertex L2 distance).
#
# Usage:
#   bash tools/corex_remote_run.sh \
#       --port-root  /path/to/libuipc-port            \
#       --vcpkg-root /root/vcpkg1                     \
#       --baseline   /path/to/nvidia-results          \
#       --output     /path/to/corex-results-rebase    \
#       [--cuda-arch ivcore11]                        \
#       [--jobs 32]
#
# Output:
#   $output/<scene>/scene_surface_*.obj
#   $output/<scene>/run.log
#   $output/diff_report.md      <- key acceptance artifact
#
# Acceptance:
#   - all 5 scenes EXIT=0, 0 sanity errors in run.log
#   - last-frame mean per-vertex L2 distance vs baseline within run-to-run noise
#     (~1e-4 m for corex float32; document anything above).

set -euo pipefail

PORT_ROOT=""
VCPKG_ROOT=""
BASELINE=""
OUTPUT=""
CUDA_ARCH="ivcore11"
JOBS="32"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port-root)  PORT_ROOT="$2"; shift 2 ;;
    --vcpkg-root) VCPKG_ROOT="$2"; shift 2 ;;
    --baseline)   BASELINE="$2"; shift 2 ;;
    --output)     OUTPUT="$2"; shift 2 ;;
    --cuda-arch)  CUDA_ARCH="$2"; shift 2 ;;
    --jobs)       JOBS="$2"; shift 2 ;;
    -h|--help)    sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

for v in PORT_ROOT VCPKG_ROOT BASELINE OUTPUT; do
  if [ -z "${!v}" ]; then
    echo "missing required --${v,,}" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT"
BUILD_DIR="$PORT_ROOT/build_corex_rebase"

echo "==> [1/3] cmake configure (-DUIPC_MUDA_USE_COREX=ON)"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
  -DUIPC_USING_LOCAL_VCPKG=ON \
  -DUIPC_MUDA_SOURCE_DIR="$PORT_ROOT/external/muda" \
  -DUIPC_WITH_CUDA_BACKEND=ON \
  -DUIPC_MUDA_USE_COREX=ON \
  -DUIPC_USE_FLOAT=ON \
  -DUIPC_COREX_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_CUDA_COMPILER="$PORT_ROOT/cmake/corex-clang-cuda-wrapper.sh" \
  -DUIPC_BUILD_EXAMPLES=ON \
  -DUIPC_BUILD_COREX_API_TESTS=ON \
  -DUIPC_BUILD_CUDA_API_PROBE=OFF \
  "$PORT_ROOT"

echo "==> [2/3] ninja -j$JOBS"
ninja -j"$JOBS"

cd "$BUILD_DIR"
DEMO="$BUILD_DIR/Release/bin/corex_demo"
if [ ! -x "$DEMO" ]; then
  echo "FATAL: $DEMO not built" >&2
  exit 3
fi

echo "==> [3/3] run 5 scenes"
declare -A SCENE_FRAMES=(
  [simple]=200
  [slope]=200
  [stack]=200
  [domino]=300
  [wrecking_ball]=400
)
declare -A SCENE_BASELINE=(
  [simple]=simple
  [slope]=slope
  [stack]=stack
  [domino]=domino
  [wrecking_ball]=wb
)
declare -A SCENE_ENV=(
  [simple]="UIPC_SIMPLE_FORCE_MU=0.4"
  [slope]=""
  [stack]=""
  [domino]="UIPC_DOMINO_TILT_DEG=12 UIPC_DOMINO_MU=0.15 UIPC_GROUND_MU=1.0"
  [wrecking_ball]=""
)

for scene in simple slope stack domino wrecking_ball; do
  out="$OUTPUT/$scene"
  mkdir -p "$out"
  envprefix="${SCENE_ENV[$scene]}"
  frames="${SCENE_FRAMES[$scene]}"
  echo "  -> $scene ($frames frames)"
  log="$out/run.log"
  (
    cd "$BUILD_DIR"
    if [ -n "$envprefix" ]; then
      env $envprefix "$DEMO" --backend cuda --scene "$scene" \
          --frames "$frames" --gpu 0 --output_dir "$out"
    else
      "$DEMO" --backend cuda --scene "$scene" \
          --frames "$frames" --gpu 0 --output_dir "$out"
    fi
  ) > "$log" 2>&1
  rc=$?
  echo "     exit=$rc errors=$(grep -ci 'error\|sanity' "$log" || true)"
done

echo "==> diff report (last-frame mean L2 vs baseline)"
REPORT="$OUTPUT/diff_report.md"
{
  echo "# corex rebase diff report"
  echo
  echo "Generated: $(date -u +%FT%TZ)"
  echo "Baseline:  $BASELINE"
  echo "Rebase:    $OUTPUT"
  echo
  echo "| Scene | Frames | Exit | Errors | Last-frame mean L2 (m) | Last-frame max L2 (m) |"
  echo "|---|---|---|---|---|---|"
  for scene in simple slope stack domino wrecking_ball; do
    frames="${SCENE_FRAMES[$scene]}"
    baseline_subdir="${SCENE_BASELINE[$scene]}"
    last=$(printf "scene_surface_%04d.obj" $((frames-1)))
    a="$BASELINE/$baseline_subdir/$last"
    b="$OUTPUT/$scene/$last"
    log="$OUTPUT/$scene/run.log"
    rc_line=$(grep -E 'exit=|EXIT' "$log" 2>/dev/null | tail -1 || true)
    err_count=$(grep -ciE 'error|sanity_error' "$log" 2>/dev/null || echo 0)
    if [ -f "$a" ] && [ -f "$b" ]; then
      stats=$(python3 - "$a" "$b" <<'PY'
import sys
a, b = sys.argv[1], sys.argv[2]
def load(p):
    pts = []
    with open(p) as f:
        for ln in f:
            if ln.startswith('v '):
                _, x, y, z = ln.split()[:4]
                pts.append((float(x), float(y), float(z)))
    return pts
A, B = load(a), load(b)
n = min(len(A), len(B))
import math
diffs = []
for i in range(n):
    dx = A[i][0]-B[i][0]; dy = A[i][1]-B[i][1]; dz = A[i][2]-B[i][2]
    diffs.append(math.sqrt(dx*dx+dy*dy+dz*dz))
mean = sum(diffs)/len(diffs) if diffs else 0.0
mx = max(diffs) if diffs else 0.0
print(f"{mean:.3e}|{mx:.3e}|n={n}/{len(A)}vs{len(B)}")
PY
)
      mean=$(echo "$stats" | cut -d'|' -f1)
      mx=$(echo "$stats" | cut -d'|' -f2)
      echo "| $scene | $frames | EXIT? | $err_count | $mean | $mx |"
    else
      echo "| $scene | $frames | MISSING | $err_count | n/a (file missing) | n/a |"
    fi
  done
  echo
  echo "## Acceptance"
  echo
  echo "- All scenes EXIT=0 and \`Errors\`=0."
  echo "- Mean L2 ≤ ~1e-4 m on all scenes => corex behavior frozen, rebase OK."
  echo "- Mean L2 in 1e-4 – 1e-2 range => investigate sidecar dependency on origin helper changes."
  echo "- Mean L2 > 1e-2 m => regression; bisect by reverting one switcher at a time and re-running."
} > "$REPORT"

echo
echo "Done. See $REPORT"
