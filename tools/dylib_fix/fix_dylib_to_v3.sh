#!/usr/bin/env bash
# Force the dylib header inside vcpkg_installed to v3.0.1.
#
# Why this exists:
#   src/core/core/internal/engine_corex.cpp.inc uses the dylib v3+ API
#   (::dylib::library, ::dylib::decorations::os_default()).
#   The repo's vcpkg.json now pins dylib to 3.0.1, but if vcpkg's
#   baseline doesn't yet know 3.0.1, OR the binary cache still has
#   2.2.1, OR the user added a stray include path with an older
#   header, the build will fail with:
#     'library' in 'class dylib' does not name a type
#   Run this script after `cmake configure` to deterministically
#   overwrite the header that the compiler will see.
#
# Usage:
#   ./tools/dylib_fix/fix_dylib_to_v3.sh <build_dir>
# Example:
#   ./tools/dylib_fix/fix_dylib_to_v3.sh build_corex
#
# Idempotent. Safe to run multiple times.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_HPP="${SCRIPT_DIR}/dylib_v3.0.1.hpp"
EXPECTED_MD5="ebbda41a28e62477f8f011ca5f5e85ed"

if [[ "$#" -lt 1 ]]; then
    echo "usage: $0 <build_dir> [triplet]"
    echo "       (triplet defaults to x64-linux)"
    exit 2
fi

BUILD_DIR="$1"
TRIPLET="${2:-x64-linux}"
TARGET_INC="${BUILD_DIR}/vcpkg_installed/${TRIPLET}/include/dylib.hpp"
TARGET_VER_CMAKE="${BUILD_DIR}/vcpkg_installed/${TRIPLET}/share/dylib/dylibConfigVersion.cmake"

# Sanity: source bundled header is the v3.0.1 we expect
src_md5=$(md5sum "$SRC_HPP" | awk '{print $1}')
if [[ "$src_md5" != "$EXPECTED_MD5" ]]; then
    echo "FATAL: bundled $SRC_HPP md5=$src_md5, expected $EXPECTED_MD5"
    echo "       (the in-repo dylib_v3.0.1.hpp got modified or corrupted)"
    exit 1
fi

if [[ ! -f "$TARGET_INC" ]]; then
    echo "ERROR: target file does not exist: $TARGET_INC"
    echo "       Run 'cmake configure' first so vcpkg installs dylib."
    exit 1
fi

cur_md5=$(md5sum "$TARGET_INC" | awk '{print $1}')
if [[ "$cur_md5" == "$EXPECTED_MD5" ]]; then
    echo "[fix_dylib] $TARGET_INC already v3.0.1 (md5 match), nothing to do."
else
    echo "[fix_dylib] overwriting $TARGET_INC"
    echo "[fix_dylib]   was: $cur_md5"
    echo "[fix_dylib]   new: $EXPECTED_MD5"
    cp "$SRC_HPP" "$TARGET_INC"
    # Touch so ninja knows to rebuild any object that includes dylib.hpp
    touch "$TARGET_INC"
fi

# Patch dylibConfigVersion.cmake so find_package(dylib 3.0.1) won't complain.
if [[ -f "$TARGET_VER_CMAKE" ]]; then
    if grep -q "PACKAGE_VERSION \"2.2.1\"" "$TARGET_VER_CMAKE"; then
        echo "[fix_dylib] patching $TARGET_VER_CMAKE (2.2.1 -> 3.0.1)"
        sed -i 's/PACKAGE_VERSION "2\.2\.1"/PACKAGE_VERSION "3.0.1"/g' "$TARGET_VER_CMAKE"
    fi
fi

echo "[fix_dylib] done."
