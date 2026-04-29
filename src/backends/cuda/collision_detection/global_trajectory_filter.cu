#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#include <collision_detection/global_trajectory_filter.h>
#include <collision_detection/trajectory_filter.h>
#include <contact_system/global_contact_manager.h>
#include <sim_engine.h>
#include <utils/corex_phase_profile.h>
#include <muda/check/check_cuda_errors.h>
#include <algorithm>
#include <cstdlib>

namespace uipc::backend
{
template <>
class SimSystemCreator<cuda::GlobalTrajectoryFilter>
{
  public:
    static U<cuda::GlobalTrajectoryFilter> create(cuda::SimEngine& engine)
    {
        auto contact_enable = engine.world().scene().config().find<IndexT>("contact/enable");
        if(contact_enable->view()[0])
            return make_unique<cuda::GlobalTrajectoryFilter>(engine);
        return nullptr;
    }
};
}  // namespace uipc::backend


namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(GlobalTrajectoryFilter);

namespace
{
__global__ void kernel_min_toi(int n, const Float* tois, Float* out)
{
    __shared__ Float block_min[256];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    Float local = 1.0f;
    while(i < n)
    {
        Float v = tois[i];
        if(v == 0.0f)
            v = 1.0f;
        local = min(local, v);
        i += blockDim.x * gridDim.x;
    }

    block_min[tid] = local;
    __syncthreads();
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(tid < stride)
            block_min[tid] = min(block_min[tid], block_min[tid + stride]);
        __syncthreads();
    }
    if(tid == 0)
        atomicMin(reinterpret_cast<int*>(out), __float_as_int(block_min[0]));
}
}  // namespace

void GlobalTrajectoryFilter::do_build()
{
    auto& config               = world().scene().config();
    auto  friction_enable_attr = config.find<IndexT>("contact/friction/enable");

    m_impl.friction_enabled = friction_enable_attr->view()[0];

    m_impl.global_contact_manager = require<GlobalContactManager>();

    on_init_scene([&] { m_impl.init(); });
}

void GlobalTrajectoryFilter::add_filter(TrajectoryFilter* filter)
{
    check_state(SimEngineState::BuildSystems, "add_filter()");
    UIPC_ASSERT(filter != nullptr, "Input TrajectoryFilter is nullptr.");
    m_impl.filters.register_sim_system(*filter);
}

void GlobalTrajectoryFilter::Impl::init()
{
    auto filter_view = filters.view();
    tois.resize(filter_view.size());
    min_toi.resize(1);
    h_tois.resize(filter_view.size());
    // Default to "no restriction": toi = 1.0.
    // Individual filters may reduce it when they detect upcoming impacts.
    tois.fill(1.0f);
    std::fill(h_tois.begin(), h_tois.end(), 1.0f);
}

void GlobalTrajectoryFilter::detect(Float alpha)
{
    auto profile_t0 = corex_profile::now_ms();
    int  filter_idx = 0;
    for(auto filter : m_impl.filters.view())
    {
        auto profile_filter_t0 = corex_profile::now_ms();
        DetectInfo info;
        info.m_alpha = alpha;
        filter->detect(info);
        corex_profile::log_phase("contact",
                                 "detect_filter",
                                 engine().frame(),
                                 engine().newton_iter(),
                                 filter_idx,
                                 corex_profile::now_ms() - profile_filter_t0);
        ++filter_idx;
    }
    corex_profile::log_phase("contact",
                             "detect",
                             engine().frame(),
                             engine().newton_iter(),
                             -1,
                             corex_profile::now_ms() - profile_t0);
}

void GlobalTrajectoryFilter::filter_active()
{
    auto profile_t0 = corex_profile::now_ms();
    if(m_impl.global_contact_manager->cfl_enabled())
    {
        auto is_acitive =
            m_impl.global_contact_manager->m_impl.vert_is_active_contact.view();
        is_acitive.fill(0);  // clear the active flag
    }

    int filter_idx = 0;
    for(auto filter : m_impl.filters.view())
    {
        auto profile_filter_t0 = corex_profile::now_ms();
        FilterActiveInfo info(&m_impl);
        filter->filter_active(info);
        corex_profile::log_phase("contact",
                                 "filter_active_filter",
                                 engine().frame(),
                                 engine().newton_iter(),
                                 filter_idx,
                                 corex_profile::now_ms() - profile_filter_t0);
        ++filter_idx;
    }
    corex_profile::log_phase("contact",
                             "filter_active",
                             engine().frame(),
                             engine().newton_iter(),
                             -1,
                             corex_profile::now_ms() - profile_t0);
}

Float GlobalTrajectoryFilter::Impl::filter_toi(Float alpha)
{
    auto profile_t0 = corex_profile::now_ms();
    // Reset tois for this evaluation. Some filters may early-out and not write toi,
    // so we must keep a valid default (1.0) to avoid bogus min-toi=0.
    tois.fill(1.0f);

    auto filter_view = filters.view();
    for(auto&& [i, filter] : enumerate(filter_view))
    {
        auto profile_filter_t0 = corex_profile::now_ms();
        FilterTOIInfo info;
        info.m_toi   = muda::VarView<Float>{tois.data() + i};
        info.m_alpha = alpha;
        filter->filter_toi(info);
        corex_profile::log_phase("contact",
                                 "filter_toi_filter",
                                 -1,
                                 -1,
                                 i,
                                 corex_profile::now_ms() - profile_filter_t0);
    }
    Float h_min_toi = 1.0f;

    if constexpr(uipc::RUNTIME_CHECK)
    {
        tois.view().copy_to(h_tois.data());
        for(auto&& [i, toi] : enumerate(h_tois))
        {
            // Some filters may output toi=0 when no candidates exist (meaning "no restriction").
            // Treat 0 as 1 to keep the global min-toi meaningful; still reject negative values.
            UIPC_ASSERT(toi >= 0.0f, "Invalid toi[{}] value: {}", filter_view[i]->name(), toi);
            if(toi == 0.0f)
                toi = 1.0f;
        }
        h_min_toi = *std::min_element(h_tois.begin(), h_tois.end());
    }
    else if(std::getenv("UIPC_COREX_TOI_DEVICE_MIN") == nullptr)
    {
        tois.view().copy_to(h_tois.data());
        h_min_toi = *std::min_element(h_tois.begin(), h_tois.end());
    }
    else
    {
        Float init = 1.0f;
        checkCudaErrors(cudaMemcpy(min_toi.data(), &init, sizeof(Float), cudaMemcpyHostToDevice));
        constexpr int block = 256;
        int n = static_cast<int>(filter_view.size());
        int grid = std::max(1, std::min((n + block - 1) / block, 8));
        kernel_min_toi<<<grid, block>>>(n, tois.data(), min_toi.data());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaMemcpy(&h_min_toi, min_toi.data(), sizeof(Float), cudaMemcpyDeviceToHost));
    }

    auto ret = h_min_toi < 1.0 ? h_min_toi : 1.0;
    corex_profile::log_phase("contact",
                             "filter_toi",
                             -1,
                             -1,
                             -1,
                             corex_profile::now_ms() - profile_t0);
    return ret;
}

Float GlobalTrajectoryFilter::filter_toi(Float alpha)
{
    return m_impl.filter_toi(alpha);
}

void GlobalTrajectoryFilter::record_friction_candidates()
{
    // Check if friction candidates should be discarded before recording new ones
    // ref: https://github.com/spiriMirror/libuipc/issues/303
    if(m_impl.should_discard_friction_candidates)
    {
        clear_friction_candidates();
        m_impl.should_discard_friction_candidates = false;
        // No need to record friction candidates if they are discarded
        // Just early return
        return;
    }

    for(auto filter : m_impl.filters.view())
    {
        RecordFrictionCandidatesInfo info;
        filter->record_friction_candidates(info);
    }
}

void GlobalTrajectoryFilter::label_active_vertices()
{
    for(auto filter : m_impl.filters.view())
    {
        LabelActiveVerticesInfo info(&m_impl);
        filter->label_active_vertices(info);
    }
}

void GlobalTrajectoryFilter::clear_friction_candidates()
{
    for(auto filter : m_impl.filters.view())
    {
        filter->clear_friction_candidates();
    }
}

void GlobalTrajectoryFilter::require_discard_friction()
{
    m_impl.should_discard_friction_candidates = true;
}

muda::BufferView<IndexT> GlobalTrajectoryFilter::LabelActiveVerticesInfo::vert_is_active() const noexcept
{
    return m_impl->global_contact_manager->m_impl.vert_is_active_contact.view();
}

void GlobalTrajectoryFilter::do_apply_recover(RecoverInfo& info)
{
    // Friction candidates are already recovered, no need to discard them.
    m_impl.should_discard_friction_candidates = false;
}
}  // namespace uipc::backend::cuda
#else
#include <collision_detection/global_trajectory_filter.h>
#include <collision_detection/trajectory_filter.h>
#include <contact_system/global_contact_manager.h>
#include <sim_engine.h>

namespace uipc::backend
{
template <>
class SimSystemCreator<cuda::GlobalTrajectoryFilter>
{
  public:
    static U<cuda::GlobalTrajectoryFilter> create(cuda::SimEngine& engine)
    {
        auto contact_enable = engine.world().scene().config().find<IndexT>("contact/enable");
        if(contact_enable->view()[0])
            return make_unique<cuda::GlobalTrajectoryFilter>(engine);
        return nullptr;
    }
};
}  // namespace uipc::backend


namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(GlobalTrajectoryFilter);

void GlobalTrajectoryFilter::do_build()
{
    auto& config               = world().scene().config();
    auto  friction_enable_attr = config.find<IndexT>("contact/friction/enable");

    m_impl.friction_enabled = friction_enable_attr->view()[0];

    m_impl.global_contact_manager = require<GlobalContactManager>();

    on_init_scene([&] { m_impl.init(); });
}

void GlobalTrajectoryFilter::add_filter(TrajectoryFilter* filter)
{
    check_state(SimEngineState::BuildSystems, "add_filter()");
    UIPC_ASSERT(filter != nullptr, "Input TrajectoryFilter is nullptr.");
    m_impl.filters.register_sim_system(*filter);
}

void GlobalTrajectoryFilter::Impl::init()
{
    auto filter_view = filters.view();
    tois.resize(filter_view.size());
    h_tois.resize(filter_view.size());
}

void GlobalTrajectoryFilter::detect(Float alpha)
{
    for(auto filter : m_impl.filters.view())
    {
        DetectInfo info;
        info.m_alpha = alpha;
        filter->detect(info);
    }
}

void GlobalTrajectoryFilter::filter_active()
{
    if(m_impl.global_contact_manager->cfl_enabled())
    {
        auto is_acitive =
            m_impl.global_contact_manager->m_impl.vert_is_active_contact.view();
        is_acitive.fill(0);  // clear the active flag
    }

    for(auto filter : m_impl.filters.view())
    {
        FilterActiveInfo info(&m_impl);
        filter->filter_active(info);
    }
}

Float GlobalTrajectoryFilter::Impl::filter_toi(Float alpha)
{
    auto filter_view = filters.view();
    for(auto&& [i, filter] : enumerate(filter_view))
    {
        FilterTOIInfo info;
        info.m_toi   = muda::VarView<Float>{tois.data() + i};
        info.m_alpha = alpha;
        filter->filter_toi(info);
    }
    tois.view().copy_to(h_tois.data());

    if constexpr(uipc::RUNTIME_CHECK)
    {
        for(auto&& [i, toi] : enumerate(h_tois))
        {
            UIPC_ASSERT(toi > 0.0f, "Invalid toi[{}] value: {}", filter_view[i]->name(), toi);
        }
    }

    auto min_toi = *std::min_element(h_tois.begin(), h_tois.end());

    return min_toi < 1.0 ? min_toi : 1.0;
}

Float GlobalTrajectoryFilter::filter_toi(Float alpha)
{
    return m_impl.filter_toi(alpha);
}

void GlobalTrajectoryFilter::record_friction_candidates()
{
    // Check if friction candidates should be discarded before recording new ones
    // ref: https://github.com/spiriMirror/libuipc/issues/303
    if(m_impl.should_discard_friction_candidates)
    {
        clear_friction_candidates();
        m_impl.should_discard_friction_candidates = false;
        // No need to record friction candidates if they are discarded
        // Just early return
        return;
    }

    for(auto filter : m_impl.filters.view())
    {
        RecordFrictionCandidatesInfo info;
        filter->record_friction_candidates(info);
    }
}

void GlobalTrajectoryFilter::label_active_vertices()
{
    for(auto filter : m_impl.filters.view())
    {
        LabelActiveVerticesInfo info(&m_impl);
        filter->label_active_vertices(info);
    }
}

void GlobalTrajectoryFilter::clear_friction_candidates()
{
    for(auto filter : m_impl.filters.view())
    {
        filter->clear_friction_candidates();
    }
}

void GlobalTrajectoryFilter::require_discard_friction()
{
    m_impl.should_discard_friction_candidates = true;
}

muda::BufferView<IndexT> GlobalTrajectoryFilter::LabelActiveVerticesInfo::vert_is_active() const noexcept
{
    return m_impl->global_contact_manager->m_impl.vert_is_active_contact.view();
}

void GlobalTrajectoryFilter::do_apply_recover(RecoverInfo& info)
{
    // Friction candidates are already recovered, no need to discard them.
    m_impl.should_discard_friction_candidates = false;
}
}  // namespace uipc::backend::cuda
#endif
