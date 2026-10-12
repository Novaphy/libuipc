#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export DLAN_SDK_ROOT="${DLAN_SDK_ROOT:-/usr/local/dlgpu/sdk}"
export UIPC_DLAN_ARCH="${UIPC_DLAN_ARCH:-dlgput64}"
PYTHON="${PYTHON:-python3}"
BUILD_DIR="${UIPC_DLAN_BUILD_DIR:-${PROJECT_ROOT}/build/dlan}"
OUTPUT_DIR="${UIPC_DLAN_TEST_OUTPUT:-${PROJECT_ROOT}/artifacts/dlan}"
test -x "${DLAN_SDK_ROOT}/bin/clang++"
test -f "${VCPKG_ROOT:?Set VCPKG_ROOT to a vcpkg checkout}/scripts/buildsystems/vcpkg.cmake"
mkdir -p "${BUILD_DIR}" "${OUTPUT_DIR}"
BUILD_DIR="$(cd "${BUILD_DIR}" && pwd)"

# The small parent project owns the manifest and builds this checkout in-place.
# No NovaPhy checkout, prebuilt libuipc, or cross-repository token is used.
"${PYTHON}" scripts/gen_vcpkg_json.py "${BUILD_DIR}" \
    --with_cuda_backend=ON --with_urdf_support=OFF --dev_mode=ON \
    > "${OUTPUT_DIR}/dependencies.log"
cmake -S tools/dlan_ci -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" \
    "-DVCPKG_MANIFEST_DIR=${BUILD_DIR}" \
    "-DUIPC_DLAN_SDK_ROOT=${DLAN_SDK_ROOT}" \
    "-DCMAKE_CUDA_COMPILER=${PROJECT_ROOT}/cmake/dlan-clang-cuda-wrapper.sh" \
    -DCMAKE_CUDA_COMPILER_ID=Clang -DCMAKE_CUDA_COMPILER_FORCED=TRUE \
    -DCMAKE_CUDA_RUNTIME_LIBRARY=None \
    "-DCUDAToolkit_ROOT=${PROJECT_ROOT}/cmake/dlan-toolkit" \
    '-DCMAKE_CUDA_FLAGS=-std=c++20 -D__CUDACC__ -D__CUDACC_VER_MAJOR__=11 -fbracket-depth=1024' \
    -DUIPC_COREX_CUDA_ARCHITECTURES=70 \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    "$@" 2>&1 | tee "${OUTPUT_DIR}/configure.log"
cmake --build "${BUILD_DIR}" --target dlan_physics \
    --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-4}" 2>&1 | tee "${OUTPUT_DIR}/build.log"
