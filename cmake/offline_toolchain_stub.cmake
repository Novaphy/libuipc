# offline_toolchain_stub.cmake
#
# A no-op CMake toolchain file used by build_offline.sh.
#
# Purpose:
#   libuipc's cmake/uipc_utils.cmake unconditionally requires
#   CMAKE_TOOLCHAIN_FILE to be set (it's intended to be the vcpkg
#   toolchain). In a fully offline scenario we are NOT using vcpkg
#   at build time -- all dependencies are already in
#   vcpkg_installed_prebuilt/ and we feed them via CMAKE_PREFIX_PATH.
#
# This stub does nothing, just exists so the variable is non-empty
# and the assert in uipc_utils.cmake passes.
