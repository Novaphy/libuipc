#!/usr/bin/env bash
set -euo pipefail

DLAN_CLANG="${DLAN_CLANG:-${DLAN_SDK_ROOT:-/usr/local/dlgpu/sdk}/bin/clang++}"
if [[ " $* " == *" __cmake_determine_cuda "* ]]; then
    printf '#$ TOP=%s\n' "$(dirname "$(dirname "${DLAN_CLANG}")")" >&2
    printf '#$ NVVMIR_LIBRARY_DIR=%s/nvvm/libdevice\n' \
        "$(dirname "$(dirname "${DLAN_CLANG}")")" >&2
    exit 0
fi

args=()
skip_next=0
for arg in "$@"; do
    if [[ "${skip_next}" == "1" ]]; then
        skip_next=0
        continue
    fi
    case "${arg}" in
        --cuda-gpu-arch=*|--offload-arch=*)
            args+=("--offload-arch=${UIPC_DLAN_ARCH:-dlgput64}") ;;
        -fgpu-rdc|-rdc=true|--relocatable-device-code=true|--expt-relaxed-constexpr|--extended-lambda|--expt-extended-lambda|-diag-suppress=*|-diag_suppress=*|--diag-suppress=*|--diag_suppress=*)
            ;;
        -Xcudafe) skip_next=1 ;;
        *) args+=("${arg}") ;;
    esac
done
exec "${DLAN_CLANG}" "${args[@]}"
