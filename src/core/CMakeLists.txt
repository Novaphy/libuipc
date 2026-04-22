find_package(Eigen3 REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(cppitertools CONFIG REQUIRED)
find_package(magic_enum CONFIG REQUIRED)
find_path(DYLIB_INCLUDE_DIRS "dylib.hpp")
# dylib v3 (corex env) ships a CONFIG package; dylib v2 (upstream NVIDIA env)
# is header-only with no CONFIG. Try the package quietly so both work; the
# linkage below uses if(TARGET ...) so absence of dylib::dylib is fine.
find_package(dylib CONFIG QUIET)
find_path(TINYGLTF_INCLUDE_DIRS "tiny_gltf.h")
find_package(cpptrace CONFIG REQUIRED)

add_library(uipc_core SHARED)
add_library(uipc::core ALIAS uipc_core)
target_compile_features(uipc_core PUBLIC cxx_std_20)

# ------------------------------------------------------------------------------
# set the compile definitions
# ------------------------------------------------------------------------------
target_compile_definitions(uipc_core PRIVATE UIPC_CORE_EXPORT_DLL=1) # export dll
if(UIPC_USE_FLOAT)
    target_compile_definitions(uipc_core PUBLIC UIPC_FLOAT_SCALAR=1)
endif()
# if debug build or release build with debug info, enable uipc runtime check
target_compile_definitions(uipc_core PUBLIC 
    $<$<CONFIG:Debug>:UIPC_RUNTIME_CHECK=1>
    $<$<CONFIG:RelWithDebInfo>:UIPC_RUNTIME_CHECK=1>
    $<$<CONFIG:Release>:UIPC_RUNTIME_CHECK=1> # release build also enable runtime check now
)
target_compile_definitions(uipc_core PUBLIC "-DUIPC_PROJECT_DIR=R\"(${PROJECT_SOURCE_DIR})\"")
target_compile_definitions(uipc_core PUBLIC "-DUIPC_VERSION_MAJOR=${UIPC_VERSION_MAJOR}")
target_compile_definitions(uipc_core PUBLIC "-DUIPC_VERSION_MINOR=${UIPC_VERSION_MINOR}")
target_compile_definitions(uipc_core PUBLIC "-DUIPC_VERSION_PATCH=${UIPC_VERSION_PATCH}")

# ------------------------------------------------------------------------------
# set the target source files
# ------------------------------------------------------------------------------
uipc_target_add_include_files(uipc_core)
target_include_directories(uipc_core PRIVATE "${DYLIB_INCLUDE_DIRS}")
target_include_directories(uipc_core PRIVATE "${TINYGLTF_INCLUDE_DIRS}")
# add the source files in the current directory to the target uipc
file(GLOB SOURCE "*.h" "*.cpp")
target_sources(uipc_core PRIVATE ${SOURCE})


# ------------------------------------------------------------------------------
# set the target link libraries
# ------------------------------------------------------------------------------
target_link_libraries(uipc_core PUBLIC 
    Eigen3::Eigen 
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    fmt::fmt
    cppitertools::cppitertools
    magic_enum::magic_enum
    cpptrace::cpptrace
)

if(TARGET dylib::dylib)
    target_link_libraries(uipc_core PUBLIC dylib::dylib)
endif()

if(UIPC_MUDA_USE_COREX)
    # Corex env: vcpkg dylib v3 is not header-only; link the compiled library
    # if available. NVIDIA env relies on dylib::dylib target (above) or pure
    # header-only dylib v2.
    if(EXISTS "${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib/libdylib.a")
        target_link_libraries(uipc_core PUBLIC "${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib/libdylib.a")
    elseif(EXISTS "/root/vcpkg/offline_installed_fresh/x64-linux/lib/libdylib.a")
        target_link_libraries(uipc_core PUBLIC "/root/vcpkg/offline_installed_fresh/x64-linux/lib/libdylib.a")
    endif()
    # Tell switcher overlays in core/{i_engine,internal/engine,internal/world,
    # sanity_checker,world}.cpp to pull in their _corex.cpp.inc sidecars
    # (dylib v3 API). NVIDIA env (UIPC_MUDA_USE_COREX=OFF) leaves the macro
    # undefined so /root/src verbatim (dylib v2 API) is compiled.
    target_compile_definitions(uipc_core PUBLIC UIPC_COREX_CUDA10_COMPAT=1)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC requires explicit linking to the dl library for dlopen, dlsym, etc.
    target_link_libraries(uipc_core PRIVATE dl)
endif()

if(MSVC)
    # define /bigobj
    target_compile_options(uipc_core PRIVATE "/bigobj")
    # nominmax to avoid conflicts on `min/max` macros on Windows
    target_compile_definitions(uipc_core PUBLIC NOMINMAX)
    # set MSVC runtime library on Windows always /MD or /MDd
    set_target_properties(uipc_core PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
    )
endif()
# ------------------------------------------------------------------------------
# add the files in the subdirectories to the target uipc
# so ensure that the uipc target is created before adding the subdirectories
# ------------------------------------------------------------------------------
add_subdirectory(common)
add_subdirectory(backend)
add_subdirectory(geometry)
add_subdirectory(constitution)
add_subdirectory(builtin)
add_subdirectory(core)
add_subdirectory(diff_sim)

# ------------------------------------------------------------------------------
# set the output directory
# ------------------------------------------------------------------------------
uipc_target_set_relative_source_file(uipc_core)
uipc_target_set_output_directory(uipc_core)
uipc_target_set_rpath(uipc_core)


# ------------------------------------------------------------------------------
# setup source group for the IDE
# ------------------------------------------------------------------------------
file(GLOB_RECURSE SOURCE_GROUP_FILES "*.h" "*.cpp" "*.inl")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/.." FILES ${SOURCE_GROUP_FILES})

# ------------------------------------------------------------------------------
# Install targets
# ------------------------------------------------------------------------------
uipc_install_target(uipc_core)











