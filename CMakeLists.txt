cmake_minimum_required(VERSION 3.25)
# set cmake include path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(cmake/uipc_utils.cmake)

# =========================================================================
# Project Version
# =========================================================================
if(DEFINED SKBUILD_PROJECT_VERSION)
   set(UIPC_VERSION ${SKBUILD_PROJECT_VERSION})
else()
    set(UIPC_VERSION "0.9.0")
endif()
uipc_parse_version(${UIPC_VERSION} UIPC_VERSION_MAJOR UIPC_VERSION_MINOR UIPC_VERSION_PATCH)

# =========================================================================
# Project Options
# =========================================================================
option(UIPC_USING_LOCAL_VCPKG "Using local vcpkg" ON)
option(UIPC_BUILD_GUI "Build GUI, turn it off if you're server only" OFF)
option(UIPC_BUILD_PYBIND "Build pyuipc" OFF)
option(UIPC_BUILD_PYTHON_WHEEL "Wheel build mode for Python post-build helper" OFF)
option(UIPC_BUILD_EXAMPLES "Build examples" ON)
option(UIPC_BUILD_TESTS "Build tests" ON)
option(UIPC_BUILD_BENCHMARKS "Build benchmarks" ON)
option(UIPC_DEV_MODE "Enable developer mode" OFF)
option(UIPC_WITH_USD_SUPPORT "Build with Universal Scene Description (USD) support" OFF)
option(UIPC_WITH_VDB_SUPPORT "Build with Volumetric DataBase (VDB) support" OFF)
# Use float as uipc::Float everywhere (Eigen, CUDA, IPC). Recommended on Iluvatar/CoreX
# where device double arithmetic is unreliable; default remains double for NVIDIA parity.
option(UIPC_USE_FLOAT "Build libuipc with Float=float (single precision)" OFF)

set(UIPC_PYTHON_EXECUTABLE_PATH "" CACHE PATH "Python executable path")
set(UIPC_USD_INSTALL_DIR "" CACHE PATH
        "USD installation directory: the folder that contains pxrConfig.cmake, e.g. ~/Software/Usd")

# Backend Options
# =========================================================================
option(UIPC_WITH_CUDA_BACKEND "Build with CUDA backend" ON)
# if MAC OS, disable CUDA backend
if(APPLE)
    # Force OFF
    set(UIPC_WITH_CUDA_BACKEND OFF CACHE BOOL "Build with CUDA backend" FORCE)
endif()

set(UIPC_CUDA_ARCHITECTURES "native" CACHE STRING "Setup CUDA architectures (e.g. \"75,89\" or \"native\")")
set(UIPC_MUDA_SOURCE_DIR "" CACHE PATH "Path to an external muda source tree")
option(UIPC_MUDA_USE_COREX "Enable Corex-specific CUDA build compatibility mode" OFF)
set(UIPC_COREX_CUDA_STANDARD "20" CACHE STRING "CUDA C++ standard used when UIPC_MUDA_USE_COREX=ON (20 required for std::span/ranges in headers)")
set(UIPC_COREX_CUDA_ARCHITECTURES "ivcore11" CACHE STRING "CUDA architectures used when UIPC_MUDA_USE_COREX=ON")
set(UIPC_COREX_CUDA_COMPILER "" CACHE FILEPATH "Path to Corex CUDA compiler (optional)")
# Iluvatar Corex 4.4 ships cudaMallocAsync / cudaFreeAsync (verified at runtime on
# Iluvatar MR-V100). Enable the async-memory path under the Corex compat branch
# instead of falling back to synchronous cudaMalloc + multi-step fallbacks.
# Set to OFF to restore the legacy synchronous path used on Corex 3.x / pre-4.4.
option(UIPC_COREX_USE_ASYNC_MEMORY "Use cudaMallocAsync/cudaFreeAsync on Corex (>= 4.4)" ON)
option(UIPC_ENABLE_GIPC_NATIVE_CONTACT "Enable native GPU_IPC-style Gauss-Newton contact Hessian assembly in CUDA contact kernels" OFF)
option(UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE "Apply classified dytopo/contact diagonal Hessians through matrix-free SpMV instead of global BCOO assembly" OFF)

if(UIPC_MUDA_USE_COREX)
    if(UIPC_COREX_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "${UIPC_COREX_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures" FORCE)
    endif()

    if(UIPC_COREX_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER "${UIPC_COREX_CUDA_COMPILER}" CACHE FILEPATH "CUDA compiler" FORCE)
    elseif(DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "$ENV{CUDA_PATH}/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
    endif()

    # Corex clang CUDA frontend is not compatible with RDC mode.
    set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF CACHE BOOL "Enable CUDA separable compilation" FORCE)

    # GCC10 + oneTBB on this environment may fail with PSTL's legacy tbb::task backend.
    # Force serial PSTL backend for build-only compatibility.
    add_compile_definitions(_PSTL_PAR_BACKEND_SERIAL=1 _GLIBCXX_USE_TBB_PAR_BACKEND=0)

    # Project-wide Corex compatibility macro. ALL targets must see this consistently
    # because shared headers in include/ have #if defined(UIPC_COREX_CUDA10_COMPAT)
    # ABI switches; mismatched definitions across targets would cause silent ODR
    # violations and link errors.
    add_compile_definitions(UIPC_COREX_CUDA10_COMPAT)
endif()

# Propagate UIPC_CUDA_ARCHITECTURES to CMake's CUDA variable so generators (VS, etc.)
# produce consistent device code settings instead of falling back to legacy defaults.
# When UIPC_MUDA_USE_COREX is ON, CMAKE_CUDA_ARCHITECTURES is already set from UIPC_COREX_CUDA_ARCHITECTURES above;
# do not overwrite it with UIPC_CUDA_ARCHITECTURES (which defaults to "native" and breaks Corex clang).
if(NOT UIPC_MUDA_USE_COREX)
    if(DEFINED UIPC_CUDA_ARCHITECTURES AND NOT UIPC_CUDA_ARCHITECTURES STREQUAL "")
        string(REPLACE "," ";" _uipc_cuda_archs "${UIPC_CUDA_ARCHITECTURES}")
        set(CMAKE_CUDA_ARCHITECTURES ${_uipc_cuda_archs} CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "Propagated UIPC_CUDA_ARCHITECTURES -> CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endif()

# Install Options
# =========================================================================
set(UIPC_INSTALL_DIR "." CACHE PATH "Install directory, default is current directory")
if(DEFINED SKBUILD)
    if(NOT SKBUILD EQUAL 2)
        uipc_error("MUST use scikit-build-core(2), but got ${SKBUILD}")
    endif()
    set(UIPC_INSTALL_DIR "uipc/_native")
    set(UIPC_BUILD_PYTHON_WHEEL ON)
endif()

if(UIPC_BUILD_PYTHON_WHEEL)
    set(UIPC_BUILD_PYBIND ON)
endif()

# =========================================================================
# Show Logo and Options
# =========================================================================
uipc_show_logo()
uipc_find_python_executable_path()
uipc_show_options()
uipc_config_vcpkg_install()


# =========================================================================
#
# Libuipc Project
#
# =========================================================================
# Enable CUDA in the top-level project when the CUDA backend is on so CMake defines
# CMAKE_CUDA_COMPILE_OBJECT / link rules for targets that compile .cu (e.g. libuipc_backend_cuda).
# Subdirectory muda also uses CUDA, but relying on it alone can leave the root generator incomplete (CMake 3.25 + Clang CUDA).
if(UIPC_WITH_CUDA_BACKEND AND NOT APPLE)
    # C is required for external targets (e.g. GKlib/METIS); CXX+CUDA alone leaves CMAKE_C_* unset.
    project(uipc VERSION ${UIPC_VERSION} LANGUAGES C CXX CUDA)
else()
    project(uipc VERSION ${UIPC_VERSION})
endif()

if(UIPC_MUDA_USE_COREX AND CMAKE_CUDA_COMPILER_ID STREQUAL "ILUVATAR")
    # CMake's ILUVATAR CUDA rule drops depfile flags under Ninja, so every .cu object
    # looks perpetually dirty. Re-inject the standard depfile placeholders to restore
    # incremental rebuilds.
    set(CMAKE_CUDA_COMPILE_OBJECT
        "<CMAKE_CUDA_COMPILER> ${_CMAKE_CUDA_EXTRA_FLAGS} <DEFINES> <INCLUDES> <FLAGS> -MD -MT <DEP_TARGET> -MF <DEP_FILE> ${_CMAKE_COMPILE_AS_CUDA_FLAG} <CUDA_COMPILE_MODE> <SOURCE> -o <OBJECT>")
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Generate compile_commands.json

# dependencies
add_subdirectory(external)

# uipc
add_subdirectory(src)

# apps
add_subdirectory(apps)

# tools (CUDA API inventory script + optional corex_cuda_api_probe)
add_subdirectory(tools)

# install headers (libuipc/include)
uipc_install_headers()
