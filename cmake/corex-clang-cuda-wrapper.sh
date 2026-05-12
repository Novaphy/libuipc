#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REAL_COMPILER_DEFAULT="/usr/local/corex/bin/clang++"
REAL_COMPILER="${REAL_COMPILER:-$REAL_COMPILER_DEFAULT}"
PRIVATE_GCC_ROOT_DEFAULT="/usr"
COREX_COMPAT_DIR_DEFAULT="$SCRIPT_DIR/corex-compat"

#
# CMake "compiler launcher" convention:
#   <launcher> <real-compiler> <compiler-args...>
# When used as CMAKE_CUDA_COMPILER_LAUNCHER, the first argument will be the
# compiler path. Support both modes:
#   - direct wrapper-as-compiler:   wrapper <args...>
#   - wrapper-as-launcher:          wrapper <compiler> <args...>
#
if [[ "${1:-}" == /*clang++* || "${1:-}" == */clang++ || "${1:-}" == */clang ]]; then
    REAL_COMPILER="$1"
    shift || true
fi

# CMake may probe the compiler with no input files through the launcher.
# Forward a harmless query instead of erroring out.
if [[ "$#" -eq 0 ]]; then
    exec "$REAL_COMPILER" --version
fi

args=()
has_x_flag=0
has_ivcore_lang=0
is_cuda_input=0

skip_next=0
for arg in "$@"; do
    if [[ "$skip_next" -eq 1 ]]; then
        skip_next=0
        continue
    fi

    # Corex clang does not accept nvcc's '-rdc=true' (unknown argument).
    if [[ "$arg" == "-rdc=true" ]]; then
        continue
    fi
    if [[ "$arg" == "--display_error_number" ]]; then
        continue
    fi
    if [[ "$arg" == "-Xcudafe" ]]; then
        skip_next=1
        continue
    fi

    if [[ "$arg" == "-x" ]]; then
        has_x_flag=1
    elif [[ "$has_x_flag" -eq 1 ]]; then
        if [[ "$arg" == "ivcore" ]]; then
            has_ivcore_lang=1
        fi
        has_x_flag=0
    fi

    if [[ "$arg" == *.cu ]]; then
        is_cuda_input=1
    fi

    args+=("$arg")
done

if [[ "$is_cuda_input" -eq 1 && "$has_ivcore_lang" -eq 0 ]]; then
    args=("-x" "ivcore" "${args[@]}")
fi

# Prefer compat CRT header shim before Corex's own include path.
COREX_COMPAT_DIR="${COREX_COMPAT_DIR:-$COREX_COMPAT_DIR_DEFAULT}"
if [[ -d "$COREX_COMPAT_DIR" ]]; then
    args=("-I" "$COREX_COMPAT_DIR" "${args[@]}")
    if [[ -f "$COREX_COMPAT_DIR/crt/host_defines.h" ]]; then
        args=("-include" "$COREX_COMPAT_DIR/crt/host_defines.h" "${args[@]}")
    fi
fi

# Ensure Corex clang can find a C++20-capable libstdc++ (e.g. <span>, <ranges>).
# Prefer an explicit env var, otherwise use the system GCC toolchain.
GCC_ROOT="${PRIVATE_GCC_ROOT:-$PRIVATE_GCC_ROOT_DEFAULT}"
if [[ -d "$GCC_ROOT" ]]; then
    args=("--gcc-toolchain=$GCC_ROOT" "${args[@]}")
fi

# Some Corex clang-CUDA setups may not automatically pick libstdc++ headers for device compilation.
# Inject system libstdc++ include paths (g++-10 preferred, then g++) when available.
GXX_BIN="${PRIVATE_GXX_BIN:-}"
if [[ -z "$GXX_BIN" ]]; then
    if command -v g++-10 >/dev/null 2>&1; then
        GXX_BIN="$(command -v g++-10)"
    elif command -v g++ >/dev/null 2>&1; then
        GXX_BIN="$(command -v g++)"
    fi
fi

if [[ -n "$GXX_BIN" ]]; then
    GCC_VER="$("$GXX_BIN" -dumpversion 2>/dev/null || true)"
    if [[ -n "$GCC_VER" ]]; then
        SYS_CXX_BASE="/usr/include/c++/$GCC_VER"
        if [[ -d "$SYS_CXX_BASE" ]]; then
            args=("-isystem" "$SYS_CXX_BASE" "${args[@]}")
            if [[ -d "$SYS_CXX_BASE/x86_64-linux-gnu" ]]; then
                args=("-isystem" "$SYS_CXX_BASE/x86_64-linux-gnu" "${args[@]}")
            fi
        fi
    fi
fi

exec "$REAL_COMPILER" "${args[@]}"
