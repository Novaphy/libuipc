#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export_nvidia_reference_contract.sh <corex_demo_bin_dir> <output_run_dir> <baseline_name>
#
# Example:
#   export_nvidia_reference_contract.sh \
#     /path/to/build/Release/bin \
#     /tmp/nvidia_simple_contract_90f \
#     ref_nvidia_simple_contract_90f

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <corex_demo_bin_dir> <output_run_dir> <baseline_name>"
  exit 1
fi

BIN_DIR="$1"
RUN_DIR="$2"
BASELINE_NAME="$3"

BASELINE_ROOT="$(cd "$(dirname "$0")/../../output/examples/corex_demo/parity_baselines" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

mkdir -p "$RUN_DIR"

pushd "$BIN_DIR" >/dev/null
timeout 300s ./corex_demo --backend cuda --scene simple --frames 90 --gpu 0 --output_dir "$RUN_DIR" > "${RUN_DIR}.log" 2>&1
popd >/dev/null

python3 "$PROJECT_ROOT/tools/simple_physics_audit/freeze_baseline.py" \
  --run-dir "$RUN_DIR" \
  --baseline-root "$BASELINE_ROOT" \
  --baseline-name "$BASELINE_NAME" \
  --label "NVIDIA simple contract 90f" \
  --source-type nvidia \
  --notes "same contract as current corex_demo simple scene"

python3 "$PROJECT_ROOT/tools/simple_physics_audit/simple_metrics.py" \
  --frames-dir "$BASELINE_ROOT/$BASELINE_NAME/frames" \
  --output "$BASELINE_ROOT/$BASELINE_NAME/metrics.json"

echo "Done: $BASELINE_ROOT/$BASELINE_NAME"
