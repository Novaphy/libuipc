add_requires("muda 09f8a0beca898b5325c7b0c1e4cf67ea4781f3b9",{system = false,configs = {with_check = true}})

target("cuda")
    add_rules("backend")
    if has_config("dev") then
        add_rules("clangd")
    end
    add_files("**.cpp", "**.cu")
    add_headerfiles("**.h", "**.inl")
    add_includedirs(os.scriptdir(), {public = true})
    if has_config("github_actions") then
        add_cugencodes("sm_89")
    else
        add_cugencodes("native")
    end
    add_rules("cuda_warning")
    add_rules("cuda_no_host_compiler_check")    
    add_cuflags("--expt-relaxed-constexpr")
    add_cuflags("--extended-lambda")
    
    add_links(
        "cudart",
        "cublas",
        "cusparse",
        "cusolver"
    )

    -- See src/backends/cuda/CMakeLists.txt for the long rationale; in
    -- short: src/backends/cuda/sanity_check/*_check.cu calls
    -- uipc::geometry::SimplicialComplexIO{}.write(path, mesh) to dump
    -- debug meshes when a sanity check fails. Those symbols live in
    -- uipc_io, not uipc_geometry. Linux/ELF tolerates the missing link
    -- edge (resolved at runtime), but Windows MSVC does not, so without
    -- this dep the xmake Windows build fails with
    --   uipc_backend_cuda.dll : fatal error LNK1120: 2 unresolved externals
    --     __imp_??0SimplicialComplexIO@geometry@uipc@@QEAA@XZ
    --     __imp_?write@SimplicialComplexIO@geometry@uipc@@QEAAX...
    -- Keep this list in lock-step with src/backends/cuda/CMakeLists.txt.
    add_deps("uipc_geometry", "uipc_io")
    on_load(function(target)
        if target:is_plat('windows') then
            target:add('defines', '__NV_NO_HOST_COMPILER_CHECK', {public = true})
            target:add('cuflags', '-allow-unsupported-compiler', {public = true})
            -- Suppress MSVC C4819 for host compilation of CUDA sources.
            target:add('cuflags', '-Xcompiler=/wd4819', {public = true})
            target:set('toolchains', 'msvc')
        end
        target:set('toolchains', 'cuda')
    end)
    add_packages("muda")
