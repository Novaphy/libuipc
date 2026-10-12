#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export DLAN_SDK_ROOT="${DLAN_SDK_ROOT:-/usr/local/dlgpu/sdk}"
"${PYTHON:-python3}" tools/dlan_ci/verify.py \
    --build-dir "${UIPC_DLAN_BUILD_DIR:-${PROJECT_ROOT}/build/dlan}" \
    --output "${UIPC_DLAN_TEST_OUTPUT:-${PROJECT_ROOT}/artifacts/dlan}"
