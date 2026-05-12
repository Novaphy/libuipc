#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#include <sim_engine.h>
#include <uipc/common/log.h>
#include <muda/muda.h>
#include <backends/common/module.h>
#include <cstdlib>
#include <string_view>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <uipc/common/timer.h>
#include <backends/common/backend_path_tool.h>
#include <uipc/backend/engine_create_info.h>

namespace uipc::backend::cuda
{
// Corex / 天数等栈上，经 muda::Launch 包一层后的首个 kernel + 默认流 sync 可能长时间阻塞；
// 使用编译器直接发射的裸 kernel + cudaDeviceSynchronize 更稳。
__global__ void uipc_cuda_engine_nop_kernel() {}

static bool corex_probe_step_enabled(std::string_view step)
{
    if(const char* raw = std::getenv("UIPC_COREX_PROBE_STEPS"))
    {
        std::string_view steps = raw;
        if(steps.empty())
            return false;
        if(steps == "all")
            return true;

        size_t begin = 0;
        while(begin < steps.size())
        {
            size_t end   = steps.find(',', begin);
            auto   token = steps.substr(begin, end == std::string_view::npos
                                                   ? std::string_view::npos
                                                   : end - begin);
            while(!token.empty() && token.front() == ' ')
                token.remove_prefix(1);
            while(!token.empty() && token.back() == ' ')
                token.remove_suffix(1);
            if(token == step)
                return true;
            if(end == std::string_view::npos)
                break;
            begin = end + 1;
        }
        return false;
    }

    // Default to the safest subset. Other probes are opt-in because they may
    // hang on problematic Corex runtime states.
    return step == "get_device";
}

static void corex_probe_marker(const char* stage, const char* marker)
{
    logger::info("[cuda] Corex probe {} {}", stage, marker);
}

static void corex_runtime_probe(const char* stage)
{
    corex_probe_marker(stage, "enter");

    if(corex_probe_step_enabled("get_device"))
    {
        corex_probe_marker(stage, "before cudaGetDevice");
        int  current_device = -1;
        auto get_device_err = cudaGetDevice(&current_device);
        logger::info("[cuda] Corex probe {} cudaGetDevice: {} (device={})",
                     stage,
                     cudaGetErrorString(get_device_err),
                     current_device);
        if(get_device_err != cudaSuccess)
            (void)cudaGetLastError();
    }

    if(corex_probe_step_enabled("flags"))
    {
        corex_probe_marker(stage, "before cudaGetDeviceFlags");
        unsigned int device_flags = 0;
        auto         flags_err    = cudaGetDeviceFlags(&device_flags);
        logger::info("[cuda] Corex probe {} cudaGetDeviceFlags: {} (flags={})",
                     stage,
                     cudaGetErrorString(flags_err),
                     device_flags);
        if(flags_err != cudaSuccess)
            (void)cudaGetLastError();
    }

    if(corex_probe_step_enabled("stream"))
    {
        cudaStream_t probe_stream = nullptr;
        corex_probe_marker(stage, "before cudaStreamCreateWithFlags");
        auto stream_err =
            cudaStreamCreateWithFlags(&probe_stream, cudaStreamNonBlocking);
        logger::info("[cuda] Corex probe {} cudaStreamCreateWithFlags: {}",
                     stage,
                     cudaGetErrorString(stream_err));
        if(stream_err == cudaSuccess)
        {
            corex_probe_marker(stage, "before cudaStreamDestroy");
            checkCudaErrors(cudaStreamDestroy(probe_stream));
        }
        else
            (void)cudaGetLastError();
    }

    if(corex_probe_step_enabled("event"))
    {
        cudaEvent_t probe_event = nullptr;
        corex_probe_marker(stage, "before cudaEventCreateWithFlags");
        auto event_err =
            cudaEventCreateWithFlags(&probe_event, cudaEventDisableTiming);
        logger::info("[cuda] Corex probe {} cudaEventCreateWithFlags: {}",
                     stage,
                     cudaGetErrorString(event_err));
        if(event_err == cudaSuccess)
        {
            corex_probe_marker(stage, "before cudaEventDestroy");
            checkCudaErrors(cudaEventDestroy(probe_event));
        }
        else
            (void)cudaGetLastError();
    }

    if(corex_probe_step_enabled("sync"))
    {
        corex_probe_marker(stage, "before cudaDeviceSynchronize");
        auto sync_err = cudaDeviceSynchronize();
        logger::info("[cuda] Corex probe {} cudaDeviceSynchronize: {}",
                     stage,
                     cudaGetErrorString(sync_err));
        (void)cudaGetLastError();
    }

    corex_probe_marker(stage, "leave");
}

void say_hello_from_cuda()
{
    uipc_cuda_engine_nop_kernel<<<1, 1>>>();
    auto launch_err = cudaGetLastError();
    if(launch_err != cudaSuccess)
    {
        logger::warn("[cuda] Corex hello kernel launch failed: {}", cudaGetErrorString(launch_err));
        (void)cudaGetLastError();
        return;
    }

    auto sync_err = cudaDeviceSynchronize();
    if(sync_err != cudaSuccess)
    {
        logger::warn("[cuda] Corex hello kernel sync failed: {}", cudaGetErrorString(sync_err));
        (void)cudaGetLastError();
    }
}

SimEngine::SimEngine(EngineCreateInfo* info)
    : backend::SimEngine(info)
{
    logger::info("[cuda] SimEngine derived ctor enter");
    try
    {
        using namespace muda;

        logger::info("Initializing Cuda Backend...");

        auto device_id = info->config["gpu"]["device"].get<IndexT>();

        // get gpu device count
        int device_count;
        logger::info("[cuda] Corex ctor: before cudaGetDeviceCount");
        checkCudaErrors(cudaGetDeviceCount(&device_count));
        logger::info("[cuda] Corex ctor: cudaGetDeviceCount -> {}", device_count);
        if(device_id >= device_count)
        {
            UIPC_WARN_WITH_LOCATION("Cannot find device with id {}. Using device 0 instead.",
                                    device_id);

            device_id = 0;
        }

        cudaDeviceProp prop;
        logger::info("[cuda] Corex ctor: before cudaGetDeviceProperties({})", device_id);
        checkCudaErrors(cudaGetDeviceProperties(&prop, device_id));
        logger::info("[cuda] Corex ctor: cudaGetDeviceProperties({}) done", device_id);
        logger::info("Device: [{}] {}", device_id, prop.name);
        logger::info("Compute Capability: {}.{}", prop.major, prop.minor);
        logger::info("Total Global Memory: {} MB", prop.totalGlobalMem / 1024 / 1024);

        corex_runtime_probe("before cudaSetDevice");

        // Required: cudaGetDevice* does not set the active device. Without this,
        // kernels / allocations may not run on the configured GPU (multi-GPU) or
        // may hit an inconsistent context on some runtimes.
        bool skip_set_device = false;
        if(const char* skip = std::getenv("UIPC_SKIP_CUDA_SET_DEVICE");
           skip && skip[0] != '\0' && skip[0] != '0')
        {
            skip_set_device = true;
            logger::info("[cuda] Corex: skipping cudaSetDevice (UIPC_SKIP_CUDA_SET_DEVICE set).");
        }
        if(!skip_set_device)
        {
            logger::info("[cuda] Corex ctor: before cudaSetDevice({})", device_id);
            checkCudaErrors(cudaSetDevice(static_cast<int>(device_id)));
            logger::info("[cuda] Corex ctor: cudaSetDevice({}) done", device_id);
        }

        corex_runtime_probe("after cudaSetDevice");

        logger::info("[cuda] Corex ctor: before Timer::set_sync_func");
        Timer::set_sync_func([] { muda::wait_device(); });
        logger::info("[cuda] Corex ctor: Timer::set_sync_func done");

        // Corex/天数：默认跳过 ctor 期预热，把首次 GPU 工作留到 world.init 的真实路径。
        // 如需显式探测 hello kernel，可手工设置 UIPC_FORCE_CUDA_HELLO_KERNEL=1。
        logger::info("[cuda] Corex ctor: before hello decision");
        if(const char* force = std::getenv("UIPC_FORCE_CUDA_HELLO_KERNEL");
           force && force[0] != '\0' && force[0] != '0')
        {
            say_hello_from_cuda();
        }
        else
        {
            logger::info("[cuda] Corex: skipping ctor hello kernel (UIPC_FORCE_CUDA_HELLO_KERNEL=1 to run it).");
        }
        logger::info("[cuda] Corex ctor: after hello decision");

        corex_runtime_probe("after ctor hello decision");

        if(const char* warmup = std::getenv("UIPC_COREX_WARMUP_ALLOC");
           warmup && warmup[0] != '\0' && warmup[0] != '0')
        {
            logger::info("[cuda] Corex ctor: warmup alloc begin");
            void* warmup_ptr = nullptr;
            auto  alloc_err  = cudaMalloc(&warmup_ptr, 256);
            logger::info("[cuda] Corex ctor: warmup cudaMalloc -> {} ptr={}",
                         cudaGetErrorString(alloc_err),
                         warmup_ptr);
            if(alloc_err == cudaSuccess)
            {
                auto free_err = cudaFree(warmup_ptr);
                logger::info("[cuda] Corex ctor: warmup cudaFree -> {}",
                             cudaGetErrorString(free_err));
            }
            else
                (void)cudaGetLastError();
            logger::info("[cuda] Corex ctor: warmup alloc end");
        }

#ifndef NDEBUG
        // if in debug mode, sync all the time to check for errors
        muda::Debug::debug_sync_all(true);
#endif
        logger::info("Cuda Backend Init Success.");
    }
    catch(const SimEngineException& e)
    {
        logger::error("Cuda Backend Init Failed: {}", e.what());
        status().push_back(core::EngineStatus::error(e.what()));
    }
}

SimEngine::~SimEngine()
{
    muda::wait_device();

    // remove the sync callback
    muda::Debug::set_sync_callback(nullptr);

    logger::info("Cuda Backend Shutdown Success.");
}

SimEngineState SimEngine::state() const noexcept
{
    return m_state;
}

void SimEngine::event_init_scene()
{
    for(auto& action : m_on_init_scene.view())
        action();
}

void SimEngine::event_rebuild_scene()
{
    for(auto& action : m_on_rebuild_scene.view())
        action();
}

void SimEngine::event_write_scene()
{
    for(auto& action : m_on_write_scene.view())
        action();
}

void SimEngine::dump_global_surface()
{
    BackendPathTool tool{workspace()};
    auto            output_folder = tool.workspace(UIPC_RELATIVE_SOURCE_FILE, "debug");
    auto            file_path = fmt::format("{}global_surface.{}.{}.{}.obj",
                                 output_folder.string(),
                                 frame(),
                                 newton_iter(),
                                 line_search_iter());

    std::vector<Vector3> positions;
    std::vector<Vector3> disps;

    auto src_ps = m_global_vertex_manager->positions();

    positions.resize(src_ps.size());
    src_ps.copy_to(positions.data());

    std::vector<Vector2i> edges;
    auto src_es = m_global_simplicial_surface_manager->surf_edges();
    edges.resize(src_es.size());
    src_es.copy_to(edges.data());

    std::vector<Vector3i> faces;
    auto src_fs = m_global_simplicial_surface_manager->surf_triangles();
    faces.resize(src_fs.size());
    src_fs.copy_to(faces.data());

    std::ofstream file(file_path);

    for(auto& pos : positions)
        file << fmt::format("v {} {} {}\n", pos.x(), pos.y(), pos.z());

    for(auto& face : faces)
        file << fmt::format("f {} {} {}\n", face.x() + 1, face.y() + 1, face.z() + 1);

    for(auto& edge : edges)
        file << fmt::format("l {} {}\n", edge.x() + 1, edge.y() + 1);

    logger::info("Dumped global surface to {}", file_path);
}

void SimEngine::dump_global_surface_pre_ccd(SizeT newton_iter)
{
    BackendPathTool tool{workspace()};
    auto            output_folder = tool.workspace(UIPC_RELATIVE_SOURCE_FILE, "debug");
    auto            file_path = fmt::format("{}global_surface.pre_ccd.{}.{}.obj",
                                 output_folder.string(),
                                 frame(),
                                 newton_iter);

    std::vector<Vector3>  global_positions;
    std::vector<Vector3>  global_displacements;
    std::vector<Vector2i> edges;
    std::vector<Vector3i> faces;

    auto src_positions = m_global_vertex_manager->positions();
    global_positions.resize(src_positions.size());
    src_positions.copy_to(global_positions.data());

    auto src_displacements = m_global_vertex_manager->displacements();
    global_displacements.resize(src_displacements.size());
    src_displacements.copy_to(global_displacements.data());

    auto src_edges = m_global_simplicial_surface_manager->surf_edges();
    edges.resize(src_edges.size());
    src_edges.copy_to(edges.data());

    auto src_faces = m_global_simplicial_surface_manager->surf_triangles();
    faces.resize(src_faces.size());
    src_faces.copy_to(faces.data());

    std::vector<Vector3> global_positions_plus_dx(global_positions.size());
    for(SizeT i = 0; i < global_positions.size(); ++i)
        global_positions_plus_dx[i] = global_positions[i] + global_displacements[i];

    std::ofstream file(file_path);

    for(const auto& pos : global_positions_plus_dx)
        file << fmt::format("v {} {} {}\n", pos.x(), pos.y(), pos.z());

    for(const auto& face : faces)
        file << fmt::format("f {} {} {}\n", face.x() + 1, face.y() + 1, face.z() + 1);

    for(const auto& edge : edges)
        file << fmt::format("l {} {}\n", edge.x() + 1, edge.y() + 1);

    logger::info("Dumped global surface to {}", file_path);
}
}  // namespace uipc::backend::cuda

// Dump & Recover:
namespace uipc::backend::cuda
{
bool SimEngine::do_dump(DumpInfo&)
{
    // Now just do nothing
    return true;
}

bool SimEngine::do_try_recover(RecoverInfo&)
{
    // Now just do nothing
    return true;
}

void SimEngine::do_apply_recover(RecoverInfo& info)
{
    // If success, set the current frame to the recovered frame
    m_current_frame = info.frame();
}

void SimEngine::do_clear_recover(RecoverInfo& info)
{
    // If failed, do nothing
}

SizeT SimEngine::get_frame() const
{
    return m_current_frame;
}

SizeT SimEngine::newton_iter() const noexcept
{
    return m_newton_iter;
}

SizeT SimEngine::line_search_iter() const noexcept
{
    return m_line_search_iter;
}
}  // namespace uipc::backend::cuda
#else
#include <sim_engine.h>
#include <uipc/common/log.h>
#include <muda/muda.h>
#include <kernel_cout.h>
#include <backends/common/module.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <uipc/common/timer.h>
#include <backends/common/backend_path_tool.h>
#include <uipc/backend/engine_create_info.h>

namespace uipc::backend::cuda
{
void say_hello_from_cuda()
{
    using namespace muda;
    Launch()
        .apply([cout = KernelCout::viewer()] __device__() mutable
               { cout << "CUDA Backend Kernel Console Init Success!\n"; })
        .wait();
}

SimEngine::SimEngine(EngineCreateInfo* info)
    : backend::SimEngine(info)
{
    try
    {
        using namespace muda;

        logger::info("Initializing Cuda Backend...");

        auto device_id = info->config["gpu"]["device"].get<IndexT>();

        // get gpu device count
        int device_count;
        checkCudaErrors(cudaGetDeviceCount(&device_count));
        if(device_id >= device_count)
        {
            UIPC_WARN_WITH_LOCATION("Cannot find device with id {}. Using device 0 instead.",
                                    device_id);

            device_id = 0;
        }

        cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, device_id));
        logger::info("Device: [{}] {}", device_id, prop.name);
        logger::info("Compute Capability: {}.{}", prop.major, prop.minor);
        logger::info("Total Global Memory: {} MB", prop.totalGlobalMem / 1024 / 1024);

        Timer::set_sync_func([] { muda::wait_device(); });

        say_hello_from_cuda();

#ifndef NDEBUG
        // if in debug mode, sync all the time to check for errors
        muda::Debug::debug_sync_all(true);
#endif
        logger::info("Cuda Backend Init Success.");
    }
    catch(const SimEngineException& e)
    {
        logger::error("Cuda Backend Init Failed: {}", e.what());
        status().push_back(core::EngineStatus::error(e.what()));
    }
}

SimEngine::~SimEngine()
{
    muda::wait_device();

    // remove the sync callback
    muda::Debug::set_sync_callback(nullptr);

    logger::info("Cuda Backend Shutdown Success.");
}

SimEngineState SimEngine::state() const noexcept
{
    return m_state;
}

void SimEngine::event_init_scene()
{
    for(auto& action : m_on_init_scene.view())
        action();
}

void SimEngine::event_rebuild_scene()
{
    for(auto& action : m_on_rebuild_scene.view())
        action();
}

void SimEngine::event_write_scene()
{
    for(auto& action : m_on_write_scene.view())
        action();
}

void SimEngine::dump_global_surface()
{
    BackendPathTool tool{workspace()};
    auto            output_folder = tool.workspace(UIPC_RELATIVE_SOURCE_FILE, "debug");
    auto            file_path = fmt::format("{}global_surface.{}.{}.{}.obj",
                                 output_folder.string(),
                                 frame(),
                                 newton_iter(),
                                 line_search_iter());

    std::vector<Vector3> positions;
    std::vector<Vector3> disps;

    auto src_ps = m_global_vertex_manager->positions();

    positions.resize(src_ps.size());
    src_ps.copy_to(positions.data());

    std::vector<Vector2i> edges;
    auto src_es = m_global_simplicial_surface_manager->surf_edges();
    edges.resize(src_es.size());
    src_es.copy_to(edges.data());

    std::vector<Vector3i> faces;
    auto src_fs = m_global_simplicial_surface_manager->surf_triangles();
    faces.resize(src_fs.size());
    src_fs.copy_to(faces.data());

    std::ofstream file(file_path);

    for(auto& pos : positions)
        file << fmt::format("v {} {} {}\n", pos.x(), pos.y(), pos.z());

    for(auto& face : faces)
        file << fmt::format("f {} {} {}\n", face.x() + 1, face.y() + 1, face.z() + 1);

    for(auto& edge : edges)
        file << fmt::format("l {} {}\n", edge.x() + 1, edge.y() + 1);

    logger::info("Dumped global surface to {}", file_path);
}

void SimEngine::dump_global_surface_pre_ccd(SizeT newton_iter)
{
    BackendPathTool tool{workspace()};
    auto            output_folder = tool.workspace(UIPC_RELATIVE_SOURCE_FILE, "debug");
    auto            file_path = fmt::format("{}global_surface.pre_ccd.{}.{}.obj",
                                 output_folder.string(),
                                 frame(),
                                 newton_iter);

    std::vector<Vector3>  global_positions;
    std::vector<Vector3>  global_displacements;
    std::vector<Vector2i> edges;
    std::vector<Vector3i> faces;

    auto src_positions = m_global_vertex_manager->positions();
    global_positions.resize(src_positions.size());
    src_positions.copy_to(global_positions.data());

    auto src_displacements = m_global_vertex_manager->displacements();
    global_displacements.resize(src_displacements.size());
    src_displacements.copy_to(global_displacements.data());

    auto src_edges = m_global_simplicial_surface_manager->surf_edges();
    edges.resize(src_edges.size());
    src_edges.copy_to(edges.data());

    auto src_faces = m_global_simplicial_surface_manager->surf_triangles();
    faces.resize(src_faces.size());
    src_faces.copy_to(faces.data());

    std::vector<Vector3> global_positions_plus_dx(global_positions.size());
    for(SizeT i = 0; i < global_positions.size(); ++i)
        global_positions_plus_dx[i] = global_positions[i] + global_displacements[i];

    std::ofstream file(file_path);

    for(const auto& pos : global_positions_plus_dx)
        file << fmt::format("v {} {} {}\n", pos.x(), pos.y(), pos.z());

    for(const auto& face : faces)
        file << fmt::format("f {} {} {}\n", face.x() + 1, face.y() + 1, face.z() + 1);

    for(const auto& edge : edges)
        file << fmt::format("l {} {}\n", edge.x() + 1, edge.y() + 1);

    logger::info("Dumped global surface to {}", file_path);
}
}  // namespace uipc::backend::cuda

// Dump & Recover:
namespace uipc::backend::cuda
{
bool SimEngine::do_dump(DumpInfo&)
{
    // Now just do nothing
    return true;
}

bool SimEngine::do_try_recover(RecoverInfo&)
{
    // Now just do nothing
    return true;
}

void SimEngine::do_apply_recover(RecoverInfo& info)
{
    // If success, set the current frame to the recovered frame
    m_current_frame = info.frame();
}

void SimEngine::do_clear_recover(RecoverInfo& info)
{
    // If failed, do nothing
}

SizeT SimEngine::get_frame() const
{
    return m_current_frame;
}

SizeT SimEngine::newton_iter() const noexcept
{
    return m_newton_iter;
}

SizeT SimEngine::line_search_iter() const noexcept
{
    return m_line_search_iter;
}
}  // namespace uipc::backend::cuda
#endif
