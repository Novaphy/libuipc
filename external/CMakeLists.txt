# 2025.10.9
# We remove all submodules from git and manage them via vcpkg instead.
# For fast-start-development, we provide a helper function to initialize submodules.
# Once the dependency is stabilized, we will make it a vcpkg dependency and remove the submodule initialization.

# Submodule Usage 
# uipc_init_submodule(my_lib) # auto init submodule if not exist
# add_subdirectory(my_lib)    # include the submodule cmake project
# set_target_properties(my_lib PROPERTIES FOLDER "external") # optional, organize in IDEs

# GKlib + METIS (bundled from Stiff-GIPC for graph partitioning)
add_subdirectory(GKlib)
add_subdirectory(METIS)
set_target_properties(GKlib PROPERTIES FOLDER "external")
set_target_properties(metis PROPERTIES FOLDER "external")

if (UIPC_WITH_CUDA_BACKEND)
    # Turn off example builds in muda
    set(MUDA_BUILD_EXAMPLE OFF CACHE BOOL "Build muda example" FORCE)

    if(UIPC_MUDA_SOURCE_DIR AND EXISTS "${UIPC_MUDA_SOURCE_DIR}/CMakeLists.txt")
        uipc_info("Using external muda source: ${UIPC_MUDA_SOURCE_DIR}")
        if(UIPC_BUILD_PYTHON_WHEEL)
            add_subdirectory("${UIPC_MUDA_SOURCE_DIR}" muda_external EXCLUDE_FROM_ALL)
        else()
            add_subdirectory("${UIPC_MUDA_SOURCE_DIR}" muda_external)
        endif()
    else()
        uipc_init_submodule(muda)
        if(UIPC_BUILD_PYTHON_WHEEL)
            # When building wheel, we don't need to install muda to the wheel package.
            add_subdirectory(muda EXCLUDE_FROM_ALL)
        else()
            add_subdirectory(muda)
        endif()
    endif()

    if(NOT TARGET muda::muda)
        if(TARGET muda)
            add_library(muda::muda ALIAS muda)
        else()
            message(FATAL_ERROR "Expected muda target from source '${UIPC_MUDA_SOURCE_DIR}', but target 'muda' was not found.")
        endif()
    endif()

    if(TARGET muda)
        set_target_properties(muda PROPERTIES FOLDER "external")

        # Keep muda interface options untouched here.
        # Corex-specific CUDA option compatibility is handled in muda/CMakeLists.txt.
    endif()
endif()
