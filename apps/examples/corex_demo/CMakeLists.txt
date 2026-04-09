include(uipc_utils)

set(app_name corex_demo)

add_executable(${app_name})
target_sources(${app_name} PRIVATE "main.cpp")
target_link_libraries(${app_name} PRIVATE uipc::uipc app)

uipc_target_set_relative_source_file(${app_name})
uipc_target_set_output_directory(${app_name})
uipc_target_add_backend_dependency(${app_name})

# Mark Corex/Days toolchain builds so main.cpp can use a lighter default scene path.
if(UIPC_MUDA_USE_COREX)
    target_compile_definitions(${app_name} PRIVATE UIPC_APP_COREX_BUILD=1)
endif()

set_property(TARGET ${app_name} PROPERTY FOLDER "apps/examples")
