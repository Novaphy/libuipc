#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#include <sim_engine.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
#include <cstdio>
#include <dytopo_effect_system/dytopo_effect_reporter.h>
#include <dytopo_effect_system/dytopo_effect_receiver.h>
#include <uipc/common/enumerate.h>
#include <kernel_cout.h>
#include <uipc/common/unit.h>
#include <uipc/common/zip.h>
#include <energy_component_flags.h>
#include <utils/corex_phase_profile.h>
#include <cstdlib>

namespace uipc::backend
{
template <>
class SimSystemCreator<cuda::GlobalDyTopoEffectManager>
{
  public:
    static U<cuda::GlobalDyTopoEffectManager> create(cuda::SimEngine& engine)
    {
        auto dytopo_effect_enable_attr =
            engine.world().scene().config().find<IndexT>("contact/enable");
        bool dytopo_effect_enable = dytopo_effect_enable_attr->view()[0] != 0;

        auto& types = engine.world().scene().constitution_tabular().types();
        bool  has_inter_primitive_constitution =
            types.find(std::string{builtin::InterPrimitive}) != types.end();

        if(dytopo_effect_enable || has_inter_primitive_constitution)
            return make_unique<cuda::GlobalDyTopoEffectManager>(engine);
        return nullptr;
    }
};
}  // namespace uipc::backend

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(GlobalDyTopoEffectManager);

muda::CBCOOVectorView<Float, 3> GlobalDyTopoEffectManager::gradients() const noexcept
{
    return m_impl.sorted_dytopo_effect_gradient.view();
}

muda::CBCOOMatrixView<Float, 3> GlobalDyTopoEffectManager::hessians() const noexcept
{
    return m_impl.sorted_dytopo_effect_hessian.view();
}

void GlobalDyTopoEffectManager::do_build()
{
    const auto& config = world().scene().config();

    m_impl.global_vertex_manager = require<GlobalVertexManager>();
}

void GlobalDyTopoEffectManager::Impl::init(WorldVisitor& world)
{
    // 3) reporters
    auto dytopo_effect_reporter_view = dytopo_effect_reporters.view();
    for(auto&& [i, R] : enumerate(dytopo_effect_reporter_view))
        R->init();
    for(auto&& [i, R] : enumerate(dytopo_effect_reporter_view))
        R->m_index = i;

    reporter_energy_offsets_counts.resize(dytopo_effect_reporter_view.size());
    reporter_gradient_offsets_counts.resize(dytopo_effect_reporter_view.size());
    reporter_hessian_offsets_counts.resize(dytopo_effect_reporter_view.size());

    // 4) receivers
    auto dytopo_effect_receiver_view = dytopo_effect_receivers.view();
    for(auto&& [i, R] : enumerate(dytopo_effect_receiver_view))
        R->init();
    for(auto&& [i, R] : enumerate(dytopo_effect_receiver_view))
        R->m_index = i;

    classified_dytopo_effect_gradients.resize(dytopo_effect_receiver_view.size());
    classified_dytopo_effect_hessians.resize(dytopo_effect_receiver_view.size());
}

void GlobalDyTopoEffectManager::Impl::compute_dytopo_effect(ComputeDyTopoEffectInfo& info)
{
    constexpr long long frame = -1;
    constexpr long long newton = -1;
    {
        corex_profile::ScopedPhase phase{"dytopo", "assemble_total", frame, newton};
        _assemble(info);
    }
    {
        corex_profile::ScopedPhase phase{"dytopo", "convert_matrix", frame, newton};
        _convert_matrix();
    }
    {
        corex_profile::ScopedPhase phase{"dytopo", "distribute_total", frame, newton};
        _distribute(info);
    }
}

void GlobalDyTopoEffectManager::Impl::_assemble(ComputeDyTopoEffectInfo& info)
{
    Timer timer{"Assemble Dytopo Effect"};

    auto vertex_count = global_vertex_manager->positions().size();

    auto reporter_gradient_counts = reporter_gradient_offsets_counts.counts();
    auto reporter_hessian_counts  = reporter_hessian_offsets_counts.counts();
    bool gradient_only            = info.m_gradient_only;

    logger::info("DyTopo Effect Assembly: GradientOnly={}, ComponentFlags={}",
                 info.m_gradient_only,
                 enum_flags_name(info.m_component_flags));

    {
        Timer timer{"Report Extent"};
        corex_profile::ScopedPhase phase{"dytopo",
                                         "report_extent",
                                         -1,
                                         -1};
        for(auto&& [i, reporter] : enumerate(dytopo_effect_reporters.view()))
        {
            reporter_gradient_counts[i] = 0;
            reporter_hessian_counts[i]  = 0;

            if(!has_flags(info.m_component_flags, reporter->component_flags()))
                continue;

            GradientHessianExtentInfo extent_info;
            extent_info.m_gradient_only = gradient_only;
            reporter->report_gradient_hessian_extent(extent_info);

            reporter_gradient_counts[i] = extent_info.m_gradient_count;
            reporter_hessian_counts[i] = gradient_only ? 0 : extent_info.m_hessian_count;
            logger::info("<{}> DyTopo Grad3 count: {}, DyTopo Hess3x3 count: {}",
                         reporter->name(),
                         extent_info.m_gradient_count,
                         extent_info.m_hessian_count);
        }
    }

    {
        Timer timer{"Scan and Allocate"};
        corex_profile::ScopedPhase phase{"dytopo",
                                         "scan_allocate",
                                         -1,
                                         -1};
        // scan
        reporter_gradient_offsets_counts.scan();
        reporter_hessian_offsets_counts.scan();

        auto total_gradient_count = reporter_gradient_offsets_counts.total_count();
        auto total_hessian_count  = reporter_hessian_offsets_counts.total_count();

        // allocate
        loose_resize_entries(collected_dytopo_effect_gradient, total_gradient_count);
        loose_resize_entries(sorted_dytopo_effect_gradient, total_gradient_count);
        loose_resize_entries(collected_dytopo_effect_hessian, total_hessian_count);
        loose_resize_entries(sorted_dytopo_effect_hessian, total_hessian_count);
        collected_dytopo_effect_gradient.reshape(vertex_count);
        collected_dytopo_effect_hessian.reshape(vertex_count, vertex_count);
    }

    // collect
    const bool profile_enabled = corex_profile::enabled();
    auto profile_collect_t0 = profile_enabled ? corex_profile::now_ms() : 0.0;
    for(auto&& [i, reporter] : enumerate(dytopo_effect_reporters.view()))
    {
        if(!has_flags(info.m_component_flags, reporter->component_flags()))
            continue;

        auto [g_offset, g_count] = reporter_gradient_offsets_counts[i];
        auto [h_offset, h_count] = reporter_hessian_offsets_counts[i];

        GradientHessianInfo info;
        info.m_gradient_only = gradient_only;

        info.m_gradients =
            collected_dytopo_effect_gradient.view().subview(g_offset, g_count);
        info.m_hessians = collected_dytopo_effect_hessian.view().subview(h_offset, h_count);

        reporter->assemble(info);
    }
    if(profile_enabled)
        corex_profile::log_phase("dytopo",
                                 "reporter_assemble",
                                 -1,
                                 -1,
                                 -1,
                                 corex_profile::now_ms() - profile_collect_t0);
}

void GlobalDyTopoEffectManager::Impl::_convert_matrix()
{
    Timer timer{"Convert Dytopo Matrix"};
    use_raw_full_gradient_distribution =
        !collected_dytopo_effect_gradient.doublet_count() ? false :
        _can_distribute_raw_full_gradient();
    use_raw_full_hessian_distribution =
        !collected_dytopo_effect_hessian.triplet_count() ? false :
        _can_distribute_raw_full_hessian();

    static const bool matrixfree_contact = []
    {
#if defined(UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE) && UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE
        const char* e = std::getenv("UIPC_GIPC_MATRIXFREE_CONTACT");
        return e && e[0] != '\0' && e[0] != '0';
#else
        return false;
#endif
    }();

    if(use_raw_full_hessian_distribution)
    {
        sorted_dytopo_effect_hessian.reshape(collected_dytopo_effect_hessian.rows(),
                                             collected_dytopo_effect_hessian.cols());
        sorted_dytopo_effect_hessian.resize_triplets(0);
    }
    else if(matrixfree_contact)
    {
        auto& from = collected_dytopo_effect_hessian;
        auto& to   = sorted_dytopo_effect_hessian;
        to.reshape(from.rows(), from.cols());
        to.resize_triplets(from.triplet_count());
        if(from.triplet_count() > 0)
        {
            to.row_indices().copy_from(from.row_indices());
            to.col_indices().copy_from(from.col_indices());
            to.values().copy_from(from.values());
        }
    }
    else
    {
        matrix_converter.convert(collected_dytopo_effect_hessian, sorted_dytopo_effect_hessian);
    }

    if(use_raw_full_gradient_distribution)
    {
        sorted_dytopo_effect_gradient.reshape(collected_dytopo_effect_gradient.count());
        sorted_dytopo_effect_gradient.resize_doublets(0);
    }
    else
    {
        matrix_converter.convert(collected_dytopo_effect_gradient, sorted_dytopo_effect_gradient);
    }
}

bool GlobalDyTopoEffectManager::Impl::_can_distribute_raw_full_gradient()
{
    auto receivers = dytopo_effect_receivers.view();
    if(receivers.size() != 1)
        return false;

    auto* receiver = receivers[0];
    if(!receiver || !receiver->accept_raw_full_gradient())
        return false;

    DyTopoClassifyInfo classify_info;
    receiver->report(classify_info);
    if(!classify_info.is_diag())
        return false;

    auto vertex_count = static_cast<IndexT>(global_vertex_manager->positions().size());
    return classify_info.gradient_i_range() == Vector2i{0, vertex_count};
}

bool GlobalDyTopoEffectManager::Impl::_can_distribute_raw_full_hessian()
{
    auto receivers = dytopo_effect_receivers.view();
    if(receivers.size() != 1)
        return false;

    auto* receiver = receivers[0];
    if(!receiver || !receiver->accept_raw_full_hessian())
        return false;

    DyTopoClassifyInfo classify_info;
    receiver->report(classify_info);
    if(!classify_info.is_diag())
        return false;

    auto vertex_count = static_cast<IndexT>(global_vertex_manager->positions().size());
    return classify_info.hessian_i_range() == Vector2i{0, vertex_count}
           && classify_info.hessian_j_range() == Vector2i{0, vertex_count};
}

void GlobalDyTopoEffectManager::Impl::_distribute(ComputeDyTopoEffectInfo& info)
{
    Timer timer{"Distribute Dytopo Effect"};

    using namespace muda;

    auto vertex_count = global_vertex_manager->positions().size();

    for(auto&& [i, receiver] : enumerate(dytopo_effect_receivers.view()))
    {
        DyTopoClassifyInfo classify_info;
        receiver->report(classify_info);

        ClassifiedDyTopoEffectInfo classified_info;
        auto& classified_gradients = classified_dytopo_effect_gradients[i];
        classified_gradients.reshape(vertex_count);
        auto& classified_hessians = classified_dytopo_effect_hessians[i];
        classified_hessians.reshape(vertex_count, vertex_count);

        // 1) report gradient
        if(use_raw_full_gradient_distribution && receiver->accept_raw_full_gradient())
        {
            classified_info.m_gradients = collected_dytopo_effect_gradient.view();
        }
        else if(classify_info.is_diag())
        {
            const auto N = sorted_dytopo_effect_gradient.doublet_count();

            // clear the range in device
            gradient_range = Vector2i{0, 0};

            // partition
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    N,
                    [gradient_range = gradient_range.viewer().name("gradient_range"),
                     dytopo_effect_gradient =
                         std::as_const(sorted_dytopo_effect_gradient).viewer().name("dytopo_effect_gradient"),
                     range = classify_info.gradient_i_range()] __device__(int I) mutable
                    {
                        auto in_range = [](int i, const Vector2i& range)
                        { return i >= range.x() && i < range.y(); };

                        auto&& [i, G]      = dytopo_effect_gradient(I);
                        bool this_in_range = in_range(i, range);

                        if(!this_in_range)
                        {
                            return;
                        }

                        bool prev_in_range = false;
                        if(I > 0)
                        {
                            auto&& [prev_i, prev_G] = dytopo_effect_gradient(I - 1);
                            prev_in_range = in_range(prev_i, range);
                        }
                        bool next_in_range = false;
                        if(I < dytopo_effect_gradient.total_doublet_count() - 1)
                        {
                            auto&& [next_i, next_G] = dytopo_effect_gradient(I + 1);
                            next_in_range = in_range(next_i, range);
                        }

                        // if the prev is not in range, then this is the start of the partition
                        if(!prev_in_range)
                        {
                            gradient_range->x() = I;
                        }
                        // if the next is not in range, then this is the end of the partition
                        if(!next_in_range)
                        {
                            gradient_range->y() = I + 1;
                        }
                    });

            Vector2i h_range = gradient_range;  // copy back

            auto count = h_range.y() - h_range.x();

            loose_resize_entries(classified_gradients, count);

            // fill
            if(count > 0)
            {
                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply(count,
                           [dytopo_effect_gradient = std::as_const(sorted_dytopo_effect_gradient)
                                                         .viewer()
                                                         .name("dytopo_effect_gradient"),
                            classified_gradient = classified_gradients.viewer().name("classified_gradient"),
                            range = h_range] __device__(int I) mutable
                           {
                               auto&& [i, G] = dytopo_effect_gradient(range.x() + I);
                               classified_gradient(I).write(i, G);
                           });
            }

            classified_info.m_gradients = classified_gradients.view();
        }

        // 2) report hessian
        if(!info.m_gradient_only && !classify_info.is_empty())
        {
            if(use_raw_full_hessian_distribution && receiver->accept_raw_full_hessian())
            {
                classified_info.m_hessians = collected_dytopo_effect_hessian.view();
                receiver->receive(classified_info);
                continue;
            }

            const auto N = sorted_dytopo_effect_hessian.triplet_count();

            // +1 for calculate the total count
            loose_resize(selected_hessian, N + 1);
            loose_resize(selected_hessian_offsets, N + 1);

            // select
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    N,
                    [selected_hessian = selected_hessian.view(0, N).viewer().name("selected_hessian"),
                     last =
                         VarView<IndexT>{selected_hessian.data() + N}.viewer().name("last"),
                     dytopo_effect_hessian =
                         sorted_dytopo_effect_hessian.cviewer().name("dytopo_effect_hessian"),
                     i_range = classify_info.hessian_i_range(),
                     j_range = classify_info.hessian_j_range()] __device__(int I) mutable
                    {
                        auto&& [i, j, H] = dytopo_effect_hessian(I);

                        auto in_range = [](int i, const Vector2i& range)
                        { return i >= range.x() && i < range.y(); };

                        selected_hessian(I) =
                            in_range(i, i_range) && in_range(j, j_range) ? 1 : 0;

                        // fill the last one as 0, so that we can calculate the total count
                        // during the exclusive scan
                        if(I == 0)
                            last = 0;
                    });

            // scan
            DeviceScan().ExclusiveSum(selected_hessian.data(),
                                      selected_hessian_offsets.data(),
                                      selected_hessian.size());

            IndexT h_total_count = 0;
            VarView<IndexT>{selected_hessian_offsets.data() + N}.copy_to(&h_total_count);

            loose_resize_entries(classified_hessians, h_total_count);

            // fill
            if(h_total_count > 0)
            {
                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply(N,
                           [selected_hessian = selected_hessian.cviewer().name("selected_hessian"),
                            selected_hessian_offsets =
                                selected_hessian_offsets.cviewer().name("selected_hessian_offsets"),
                            dytopo_effect_hessian =
                                sorted_dytopo_effect_hessian.cviewer().name("dytopo_effect_hessian"),
                            classified_hessian = classified_hessians.viewer().name("classified_hessian"),
                            i_range = classify_info.hessian_i_range(),
                            j_range = classify_info.hessian_j_range()] __device__(int I) mutable
                           {
                               if(selected_hessian(I))
                               {
                                   auto&& [i, j, H] = dytopo_effect_hessian(I);
                                   auto offset = selected_hessian_offsets(I);

                                   classified_hessian(offset).write(i, j, H);
                               }
                           });
            }

            classified_info.m_hessians = classified_hessians.view();
        }

        receiver->receive(classified_info);
    }
}

void GlobalDyTopoEffectManager::Impl::loose_resize_entries(
    muda::DeviceTripletMatrix<Float, 3>& m, SizeT size)
{
    if(size > m.triplet_capacity())
    {
        m.reserve_triplets(size * reserve_ratio);
    }
    m.resize_triplets(size);
}

void GlobalDyTopoEffectManager::Impl::loose_resize_entries(
    muda::DeviceDoubletVector<Float, 3>& v, SizeT size)
{
    if(size > v.doublet_capacity())
    {
        v.reserve_doublets(size * reserve_ratio);
    }
    v.resize_doublets(size);
}
}  // namespace uipc::backend::cuda


namespace uipc::backend::cuda
{
void GlobalDyTopoEffectManager::init()
{
    m_impl.init(world());
}

void GlobalDyTopoEffectManager::compute_dytopo_effect(ComputeDyTopoEffectInfo& info)
{
    m_impl.compute_dytopo_effect(info);
}

void GlobalDyTopoEffectManager::compute_dytopo_effect()
{
    ComputeDyTopoEffectInfo info;
    m_impl.compute_dytopo_effect(info);
}

void GlobalDyTopoEffectManager::add_reporter(DyTopoEffectReporter* reporter)
{
    check_state(SimEngineState::BuildSystems, "add_reporter()");
    UIPC_ASSERT(reporter != nullptr, "reporter is nullptr");
    auto flag = reporter->component_flags();
    UIPC_ASSERT(is_valid_flag(flag),
                "reporter component_flags() is not valid single flag, it's {}",
                enum_flags_name(flag));
    m_impl.dytopo_effect_reporters.register_sim_system(*reporter);

    // classify into contact / non-contact
    if(reporter->component_flags() == EnergyComponentFlags::Contact)
    {
        m_impl.contact_reporters.register_sim_system(*reporter);
    }
    else
    {
        m_impl.non_contact_reporters.register_sim_system(*reporter);
    }
}

void GlobalDyTopoEffectManager::add_receiver(DyTopoEffectReceiver* receiver)
{
    check_state(SimEngineState::BuildSystems, "add_receiver()");
    UIPC_ASSERT(receiver != nullptr, "receiver is nullptr");
    m_impl.dytopo_effect_receivers.register_sim_system(*receiver);
}
}  // namespace uipc::backend::cuda

// ============================================================================
// CoreX matrix converter kernel implementations
// Placed here (existing .cu file) because CoreX CUDA runtime fails to register
// device code from newly-added .cu files in dynamically loaded shared libraries.
// ============================================================================
#include <algorithm/corex_matrix_converter_kernels.h>
#include <cstdlib>
#include <vector>

namespace uipc::backend::cuda::corex_matconv
{
using uipc::Float;

namespace
{
inline bool corex_matconv_trace()
{
    static const bool enabled = std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM") != nullptr;
    return enabled;
}

inline void corex_matconv_sync_if_needed(const char* name)
{
    if(!corex_matconv_trace())
        return;

    auto start = corex_profile::now_ms();
    cudaDeviceSynchronize();
    corex_profile::log_phase(
        "matconv_sync", name, -1, -1, -1, corex_profile::now_ms() - start);
}

class MatconvPhase
{
  public:
    explicit MatconvPhase(const char* name)
        : m_name(name)
        , m_enabled(corex_profile::enabled())
        , m_start(m_enabled ? corex_profile::now_ms() : 0.0)
    {
    }

    ~MatconvPhase()
    {
        if(m_enabled)
            corex_profile::log_phase(
                "matconv_kernel", m_name, -1, -1, -1, corex_profile::now_ms() - m_start);
    }

  private:
    const char* m_name;
    bool        m_enabled;
    double      m_start;
};
}  // namespace

static __global__ void kernel_hash_ij(int N, const int* row_indices, const int* col_indices,
                                      uint64_t* ij_hash, int* sort_index)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    ij_hash[i] = (static_cast<uint64_t>(row_indices[i]) << 32)
                 + static_cast<uint64_t>(col_indices[i]);
    sort_index[i] = i;
}

static __global__ void kernel_hash_ij_compact(int N, const int* row_indices,
                                              const int* col_indices, int col_count,
                                              uint64_t* ij_hash, int* sort_index)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    ij_hash[i] = static_cast<uint64_t>(row_indices[i]) * static_cast<uint64_t>(col_count)
                 + static_cast<uint64_t>(col_indices[i]);
    sort_index[i] = i;
}

static __global__ void kernel_decode_hash(int N, const uint64_t* ij_hash, int2* ij_pairs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    ij_pairs[i].x = static_cast<int>(ij_hash[i] >> 32);
    ij_pairs[i].y = static_cast<int>(ij_hash[i] & 0xFFFFFFFF);
}

static __global__ void kernel_decode_hash_compact(int N, const uint64_t* ij_hash,
                                                  int col_count, int2* ij_pairs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    const uint64_t hash = ij_hash[i];
    ij_pairs[i].x = static_cast<int>(hash / static_cast<uint64_t>(col_count));
    ij_pairs[i].y = static_cast<int>(hash % static_cast<uint64_t>(col_count));
}

static __global__ void kernel_write_unique_ij(int N, const int2* unique_ij_pairs,
                                              int* row_indices, int* col_indices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    row_indices[i] = unique_ij_pairs[i].x;
    col_indices[i] = unique_ij_pairs[i].y;
}

static __global__ void kernel_mark_partition(int N, const int* unique_counts,
                                             const int* offsets, int* sorted_partition)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    sorted_partition[offsets[i] + unique_counts[i] - 1] = 1;
}

static __global__ void kernel_fill_segment_ids_from_offsets(int N,
                                                            const int* unique_counts,
                                                            const int* offsets,
                                                            int* segment_ids)
{
    int seg = blockIdx.x;
    if(seg >= N) return;

    const int begin = offsets[seg];
    const int count = unique_counts[seg];
    for(int j = threadIdx.x; j < count; j += blockDim.x)
        segment_ids[begin + j] = seg;
}

static __global__ void kernel_write_unique_indices(int N, const int* unique_indices,
                                                   int* dst_indices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    dst_indices[i] = unique_indices[i];
}

static __global__ void kernel_scatter_col_counts(int N, const int* unique_indices,
                                                 const int* counts, int* col_counts_per_row)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    col_counts_per_row[unique_indices[i]] = counts[i];
}

static __global__ void kernel_copy_sorted_blocks_3x3(int N, const BlockT3* src,
                                                      const int* sort_index, BlockT3* dst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    dst[i] = src[sort_index[i]];
}

static __global__ void kernel_copy_sorted_blocks_with_ij_3x3(
    int N, const BlockT3* src, const int* sort_index,
    const int2* ij_pairs, BlockT3* dst, int* dst_row, int* dst_col)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    dst[i]     = src[sort_index[i]];
    dst_row[i] = ij_pairs[i].x;
    dst_col[i] = ij_pairs[i].y;
}

static constexpr int kBlock = 512;
static inline int grid_for(int n) { return (n + kBlock - 1) / kBlock; }

void launch_hash_ij(int N, const int* row_indices, const int* col_indices,
                    uint64_t* ij_hash, int* sort_index)
{
    corex_matconv_sync_if_needed("hash_ij_pre");
    if(corex_matconv_trace())
    {
        cudaError_t pre2 = cudaGetLastError();
        fmt::println(stderr, "[corex_matconv] pre-err={}", cudaGetErrorString(pre2));
        std::fflush(stderr);

        fmt::println(stderr, "[corex_matconv] launching kernel_hash_ij N={} grid={} block={}", N, grid_for(N), kBlock);
        std::fflush(stderr);
    }

    MatconvPhase phase("hash_ij");
    kernel_hash_ij<<<grid_for(N), kBlock>>>(N, row_indices, col_indices, ij_hash, sort_index);

    if(corex_matconv_trace())
    {
        fmt::println(stderr, "[corex_matconv] kernel launched, checking error...");
        std::fflush(stderr);
        cudaError_t e = cudaGetLastError();
        fmt::println(stderr, "[corex_matconv] launch={}", cudaGetErrorString(e));
        std::fflush(stderr);
    }
    corex_matconv_sync_if_needed("hash_ij_post");
    if(corex_matconv_trace())
    {
        fmt::println(stderr, "[corex_matconv] sync done");
        std::fflush(stderr);
    }
}

void launch_decode_hash(int N, const uint64_t* ij_hash, int* ij_pairs_xy)
{
    MatconvPhase phase("decode_hash");
    kernel_decode_hash<<<grid_for(N), kBlock>>>(N, ij_hash, reinterpret_cast<int2*>(ij_pairs_xy));
    corex_matconv_sync_if_needed("decode_hash");
}

void launch_hash_ij_compact(int N, const int* row_indices, const int* col_indices,
                            int col_count, uint64_t* ij_hash, int* sort_index)
{
    corex_matconv_sync_if_needed("hash_ij_compact_pre");
    MatconvPhase phase("hash_ij_compact");
    kernel_hash_ij_compact<<<grid_for(N), kBlock>>>(
        N, row_indices, col_indices, col_count, ij_hash, sort_index);
    corex_matconv_sync_if_needed("hash_ij_compact_post");
}

void launch_decode_hash_compact(int N, const uint64_t* ij_hash, int col_count,
                                int* ij_pairs_xy)
{
    MatconvPhase phase("decode_hash_compact");
    kernel_decode_hash_compact<<<grid_for(N), kBlock>>>(
        N, ij_hash, col_count, reinterpret_cast<int2*>(ij_pairs_xy));
    corex_matconv_sync_if_needed("decode_hash_compact");
}

void launch_write_unique_ij(int N, const int* unique_ij_pairs_xy,
                            int* row_indices, int* col_indices)
{
    MatconvPhase phase("write_unique_ij");
    kernel_write_unique_ij<<<grid_for(N), kBlock>>>(
        N, reinterpret_cast<const int2*>(unique_ij_pairs_xy), row_indices, col_indices);
    corex_matconv_sync_if_needed("write_unique_ij");
}

void launch_mark_partition(int N, const int* unique_counts,
                           const int* offsets, int* sorted_partition)
{
    MatconvPhase phase("mark_partition");
    kernel_mark_partition<<<grid_for(N), kBlock>>>(N, unique_counts, offsets, sorted_partition);
    corex_matconv_sync_if_needed("mark_partition");
}

void launch_fill_segment_ids_from_offsets(int N, const int* unique_counts,
                                          const int* offsets, int* segment_ids)
{
    if(N <= 0)
        return;
    MatconvPhase phase("fill_segment_ids_from_offsets");
    kernel_fill_segment_ids_from_offsets<<<N, kBlock>>>(
        N, unique_counts, offsets, segment_ids);
    corex_matconv_sync_if_needed("fill_segment_ids_from_offsets");
}

void launch_write_unique_indices(int N, const int* unique_indices, int* dst_indices)
{
    MatconvPhase phase("write_unique_indices");
    kernel_write_unique_indices<<<grid_for(N), kBlock>>>(N, unique_indices, dst_indices);
    corex_matconv_sync_if_needed("write_unique_indices");
}

void launch_scatter_col_counts(int N, const int* unique_indices,
                               const int* counts, int* col_counts_per_row)
{
    MatconvPhase phase("scatter_col_counts");
    kernel_scatter_col_counts<<<grid_for(N), kBlock>>>(N, unique_indices, counts, col_counts_per_row);
    corex_matconv_sync_if_needed("scatter_col_counts");
}

void launch_copy_sorted_blocks_3x3(int N, const BlockT3* src_blocks,
                                    const int* sort_index, BlockT3* dst_blocks)
{
    MatconvPhase phase("copy_sorted_blocks_3x3");
    kernel_copy_sorted_blocks_3x3<<<grid_for(N), kBlock>>>(N, src_blocks, sort_index, dst_blocks);
    corex_matconv_sync_if_needed("copy_sorted_blocks_3x3");
}

void launch_copy_sorted_blocks_with_ij_3x3(int N, const BlockT3* src_blocks,
                                            const int* sort_index,
                                            const int* ij_pairs_xy,
                                            BlockT3* dst_blocks,
                                            int* dst_row, int* dst_col)
{
    MatconvPhase phase("copy_sorted_blocks_with_ij_3x3");
    kernel_copy_sorted_blocks_with_ij_3x3<<<grid_for(N), kBlock>>>(
        N, src_blocks, sort_index,
        reinterpret_cast<const int2*>(ij_pairs_xy),
        dst_blocks, dst_row, dst_col);
    corex_matconv_sync_if_needed("copy_sorted_blocks_with_ij_3x3");
}

__device__ __forceinline__ void corex_atomic_add_double(double* address, double val)
{
    unsigned long long* address_as_ull = reinterpret_cast<unsigned long long*>(address);
    unsigned long long  old_val = *address_as_ull;
    unsigned long long  assumed;
    do {
        assumed = old_val;
        old_val = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while(assumed != old_val);
}

static __global__ void kernel_segmental_reduce_3x3_linear(int N,
                                                          const int* segment_ids,
                                                          const BlockT3* in_blocks,
                                                          BlockT3* out_blocks,
                                                          int out_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;

    int seg = segment_ids[i];
    if(seg < 0 || seg >= out_count) return;

    const Float* src = reinterpret_cast<const Float*>(in_blocks + i);
    Float*       dst = reinterpret_cast<Float*>(out_blocks + seg);
    for(int j = 0; j < 9; ++j)
        atomicAdd(dst + j, src[j]);
}

static __global__ void kernel_segmental_reduce_3x3_blocked(int N,
                                                           const int* unique_counts,
                                                           const int* offsets,
                                                           const BlockT3* in_blocks,
                                                           BlockT3* out_blocks,
                                                           int out_count)
{
    int seg = blockIdx.x;
    if(seg >= out_count) return;

    int begin = offsets[seg];
    int count = unique_counts[seg];
    int end   = begin + count;
    if(begin < 0 || count <= 0 || begin >= N) return;
    if(end > N) end = N;

    __shared__ Float partial[256 * 9];
    Float local[9];
    for(int k = 0; k < 9; ++k)
        local[k] = Float(0);

    for(int i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        const Float* src = reinterpret_cast<const Float*>(in_blocks + i);
        for(int k = 0; k < 9; ++k)
            local[k] += src[k];
    }

    for(int k = 0; k < 9; ++k)
        partial[threadIdx.x * 9 + k] = local[k];
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(threadIdx.x < stride)
        {
            for(int k = 0; k < 9; ++k)
                partial[threadIdx.x * 9 + k] += partial[(threadIdx.x + stride) * 9 + k];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        Float* dst = reinterpret_cast<Float*>(out_blocks + seg);
        for(int k = 0; k < 9; ++k)
            dst[k] = partial[k];
    }
}

static __global__ void kernel_segmental_reduce_3x3_hybrid_small(int N,
                                                                const int* segment_ids,
                                                                const int* unique_counts,
                                                                const BlockT3* in_blocks,
                                                                BlockT3* out_blocks,
                                                                int out_count,
                                                                int threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;

    int seg = segment_ids[i];
    if(seg < 0 || seg >= out_count) return;
    if(unique_counts[seg] >= threshold) return;

    const Float* src = reinterpret_cast<const Float*>(in_blocks + i);
    Float*       dst = reinterpret_cast<Float*>(out_blocks + seg);
    for(int j = 0; j < 9; ++j)
        atomicAdd(dst + j, src[j]);
}

static __global__ void kernel_segmental_reduce_3x3_hybrid_large(int N,
                                                                const int* unique_counts,
                                                                const int* offsets,
                                                                const BlockT3* in_blocks,
                                                                BlockT3* out_blocks,
                                                                int out_count,
                                                                int threshold)
{
    int seg = blockIdx.x;
    if(seg >= out_count) return;

    int count = unique_counts[seg];
    if(count < threshold) return;

    int begin = offsets[seg];
    int end   = begin + count;
    if(begin < 0 || count <= 0 || begin >= N) return;
    if(end > N) end = N;

    __shared__ Float partial[256 * 9];
    Float local[9];
    for(int k = 0; k < 9; ++k)
        local[k] = Float(0);

    for(int i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        const Float* src = reinterpret_cast<const Float*>(in_blocks + i);
        for(int k = 0; k < 9; ++k)
            local[k] += src[k];
    }

    for(int k = 0; k < 9; ++k)
        partial[threadIdx.x * 9 + k] = local[k];
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(threadIdx.x < stride)
        {
            for(int k = 0; k < 9; ++k)
                partial[threadIdx.x * 9 + k] += partial[(threadIdx.x + stride) * 9 + k];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        Float* dst = reinterpret_cast<Float*>(out_blocks + seg);
        for(int k = 0; k < 9; ++k)
            dst[k] = partial[k];
    }
}

void launch_segmental_reduce_3x3(int N, const int* segment_ids,
                                  const BlockT3* in_blocks, BlockT3* out_blocks,
                                  int out_count)
{
    if(corex_matconv_trace())
    {
        fmt::println(stderr, "[corex_seg3x3] enter N={} out_count={}", N, out_count);
        std::fflush(stderr);
    }
    const bool profile_enabled = corex_profile::enabled();
    auto memset_start = profile_enabled ? corex_profile::now_ms() : 0.0;
    cudaMemsetAsync(out_blocks, 0, out_count * sizeof(BlockT3));
    if(profile_enabled)
        corex_profile::log_phase("matconv_kernel",
                                 "segmental_reduce_3x3_memset",
                                 -1,
                                 -1,
                                 -1,
                                 corex_profile::now_ms() - memset_start);
    corex_matconv_sync_if_needed("segmental_reduce_3x3_memset");
    if(corex_matconv_trace())
    {
        fmt::println(stderr, "[corex_seg3x3] memset+sync done");
        std::fflush(stderr);
    }
    if(out_count > 0)
    {
        MatconvPhase phase("segmental_reduce_3x3_linear");
        kernel_segmental_reduce_3x3_linear<<<grid_for(N), kBlock>>>(
            N, segment_ids, in_blocks, out_blocks, out_count);
        cudaError_t e = cudaGetLastError();
        if(corex_matconv_trace())
        {
            fmt::println(stderr, "[corex_seg3x3] kernel launch err={}", cudaGetErrorString(e));
            std::fflush(stderr);
        }
        corex_matconv_sync_if_needed("segmental_reduce_3x3");
        if(corex_matconv_trace())
        {
            fmt::println(stderr, "[corex_seg3x3] kernel sync done");
            std::fflush(stderr);
        }
    }
}

void launch_segmental_reduce_3x3_blocked(int N, const int* segment_ids,
                                         const int* unique_counts,
                                         const int* offsets,
                                         const BlockT3* in_blocks,
                                         BlockT3* out_blocks,
                                         int out_count)
{
    (void)unique_counts;
    (void)offsets;
    launch_segmental_reduce_3x3(N, segment_ids, in_blocks, out_blocks, out_count);
}

static __global__ void kernel_segmental_reduce_3x1_linear(int N,
                                                          const int* segment_ids,
                                                          const VecT3* in_vecs,
                                                          VecT3* out_vecs,
                                                          int out_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;

    int seg = segment_ids[i];
    if(seg < 0 || seg >= out_count) return;

    const Float* src = reinterpret_cast<const Float*>(in_vecs + i);
    Float*       dst = reinterpret_cast<Float*>(out_vecs + seg);
    for(int j = 0; j < 3; ++j)
        atomicAdd(dst + j, src[j]);
}

static __global__ void kernel_segmental_reduce_3x1_blocked(int N,
                                                           const int* unique_counts,
                                                           const int* offsets,
                                                           const VecT3* in_vecs,
                                                           VecT3* out_vecs,
                                                           int out_count)
{
    int seg = blockIdx.x;
    if(seg >= out_count) return;

    int begin = offsets[seg];
    int count = unique_counts[seg];
    int end   = begin + count;
    if(begin < 0 || count <= 0 || begin >= N) return;
    if(end > N) end = N;

    __shared__ Float partial[256 * 3];
    Float local[3] = {Float(0), Float(0), Float(0)};

    for(int i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        const Float* src = reinterpret_cast<const Float*>(in_vecs + i);
        for(int k = 0; k < 3; ++k)
            local[k] += src[k];
    }

    for(int k = 0; k < 3; ++k)
        partial[threadIdx.x * 3 + k] = local[k];
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(threadIdx.x < stride)
        {
            for(int k = 0; k < 3; ++k)
                partial[threadIdx.x * 3 + k] += partial[(threadIdx.x + stride) * 3 + k];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        Float* dst = reinterpret_cast<Float*>(out_vecs + seg);
        for(int k = 0; k < 3; ++k)
            dst[k] = partial[k];
    }
}

static __global__ void kernel_segmental_reduce_3x1_hybrid_small(int N,
                                                                const int* segment_ids,
                                                                const int* unique_counts,
                                                                const VecT3* in_vecs,
                                                                VecT3* out_vecs,
                                                                int out_count,
                                                                int threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;

    int seg = segment_ids[i];
    if(seg < 0 || seg >= out_count) return;
    if(unique_counts[seg] >= threshold) return;

    const Float* src = reinterpret_cast<const Float*>(in_vecs + i);
    Float*       dst = reinterpret_cast<Float*>(out_vecs + seg);
    for(int j = 0; j < 3; ++j)
        atomicAdd(dst + j, src[j]);
}

static __global__ void kernel_segmental_reduce_3x1_hybrid_large(int N,
                                                                const int* unique_counts,
                                                                const int* offsets,
                                                                const VecT3* in_vecs,
                                                                VecT3* out_vecs,
                                                                int out_count,
                                                                int threshold)
{
    int seg = blockIdx.x;
    if(seg >= out_count) return;

    int count = unique_counts[seg];
    if(count < threshold) return;

    int begin = offsets[seg];
    int end   = begin + count;
    if(begin < 0 || count <= 0 || begin >= N) return;
    if(end > N) end = N;

    __shared__ Float partial[256 * 3];
    Float local[3] = {Float(0), Float(0), Float(0)};

    for(int i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        const Float* src = reinterpret_cast<const Float*>(in_vecs + i);
        for(int k = 0; k < 3; ++k)
            local[k] += src[k];
    }

    for(int k = 0; k < 3; ++k)
        partial[threadIdx.x * 3 + k] = local[k];
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(threadIdx.x < stride)
        {
            for(int k = 0; k < 3; ++k)
                partial[threadIdx.x * 3 + k] += partial[(threadIdx.x + stride) * 3 + k];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        Float* dst = reinterpret_cast<Float*>(out_vecs + seg);
        for(int k = 0; k < 3; ++k)
            dst[k] = partial[k];
    }
}

void launch_segmental_reduce_3x1(int N, const int* segment_ids,
                                  const VecT3* in_vecs, VecT3* out_vecs,
                                  int out_count)
{
    if(corex_matconv_trace())
    {
        fmt::println(stderr, "[corex_seg3x1] enter N={} out_count={}", N, out_count);
        std::fflush(stderr);
    }
    const bool profile_enabled = corex_profile::enabled();
    auto memset_start = profile_enabled ? corex_profile::now_ms() : 0.0;
    cudaMemsetAsync(out_vecs, 0, out_count * sizeof(VecT3));
    if(profile_enabled)
        corex_profile::log_phase("matconv_kernel",
                                 "segmental_reduce_3x1_memset",
                                 -1,
                                 -1,
                                 -1,
                                 corex_profile::now_ms() - memset_start);
    corex_matconv_sync_if_needed("segmental_reduce_3x1_memset");
    if(corex_matconv_trace())
    {
        fmt::println(stderr, "[corex_seg3x1] memset+sync done");
        std::fflush(stderr);
    }
    if(out_count > 0)
    {
        MatconvPhase phase("segmental_reduce_3x1_linear");
        kernel_segmental_reduce_3x1_linear<<<grid_for(N), kBlock>>>(
            N, segment_ids, in_vecs, out_vecs, out_count);
        cudaError_t e = cudaGetLastError();
        if(corex_matconv_trace())
        {
            fmt::println(stderr, "[corex_seg3x1] kernel launch err={}", cudaGetErrorString(e));
            std::fflush(stderr);
        }
        corex_matconv_sync_if_needed("segmental_reduce_3x1");
        if(corex_matconv_trace())
        {
            fmt::println(stderr, "[corex_seg3x1] kernel sync done");
            std::fflush(stderr);
        }
    }
}

void launch_segmental_reduce_3x1_blocked(int N, const int* segment_ids,
                                         const int* unique_counts,
                                         const int* offsets,
                                         const VecT3* in_vecs,
                                         VecT3* out_vecs,
                                         int out_count)
{
    (void)unique_counts;
    (void)offsets;
    launch_segmental_reduce_3x1(N, segment_ids, in_vecs, out_vecs, out_count);
}

}  // namespace uipc::backend::cuda::corex_matconv
#else
#include <sim_engine.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
#include <dytopo_effect_system/dytopo_effect_reporter.h>
#include <dytopo_effect_system/dytopo_effect_receiver.h>
#include <uipc/common/enumerate.h>
#include <kernel_cout.h>
#include <uipc/common/unit.h>
#include <uipc/common/zip.h>
#include <energy_component_flags.h>

namespace uipc::backend
{
template <>
class SimSystemCreator<cuda::GlobalDyTopoEffectManager>
{
  public:
    static U<cuda::GlobalDyTopoEffectManager> create(cuda::SimEngine& engine)
    {
        auto dytopo_effect_enable_attr =
            engine.world().scene().config().find<IndexT>("contact/enable");
        bool dytopo_effect_enable = dytopo_effect_enable_attr->view()[0] != 0;

        auto& types = engine.world().scene().constitution_tabular().types();
        bool  has_inter_primitive_constitution =
            types.find(std::string{builtin::InterPrimitive}) != types.end();

        if(dytopo_effect_enable || has_inter_primitive_constitution)
            return make_unique<cuda::GlobalDyTopoEffectManager>(engine);
        return nullptr;
    }
};
}  // namespace uipc::backend

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(GlobalDyTopoEffectManager);

muda::CBCOOVectorView<Float, 3> GlobalDyTopoEffectManager::gradients() const noexcept
{
    return m_impl.sorted_dytopo_effect_gradient.view();
}

muda::CBCOOMatrixView<Float, 3> GlobalDyTopoEffectManager::hessians() const noexcept
{
    return m_impl.sorted_dytopo_effect_hessian.view();
}

void GlobalDyTopoEffectManager::do_build()
{
    const auto& config = world().scene().config();

    m_impl.global_vertex_manager = require<GlobalVertexManager>();
}

void GlobalDyTopoEffectManager::Impl::init(WorldVisitor& world)
{
    // 3) reporters
    auto dytopo_effect_reporter_view = dytopo_effect_reporters.view();
    for(auto&& [i, R] : enumerate(dytopo_effect_reporter_view))
        R->init();
    for(auto&& [i, R] : enumerate(dytopo_effect_reporter_view))
        R->m_index = i;

    reporter_energy_offsets_counts.resize(dytopo_effect_reporter_view.size());
    reporter_gradient_offsets_counts.resize(dytopo_effect_reporter_view.size());
    reporter_hessian_offsets_counts.resize(dytopo_effect_reporter_view.size());

    // 4) receivers
    auto dytopo_effect_receiver_view = dytopo_effect_receivers.view();
    for(auto&& [i, R] : enumerate(dytopo_effect_receiver_view))
        R->init();
    for(auto&& [i, R] : enumerate(dytopo_effect_receiver_view))
        R->m_index = i;

    classified_dytopo_effect_gradients.resize(dytopo_effect_receiver_view.size());
    classified_dytopo_effect_hessians.resize(dytopo_effect_receiver_view.size());
}

void GlobalDyTopoEffectManager::Impl::compute_dytopo_effect(ComputeDyTopoEffectInfo& info)
{
    _assemble(info);
    _convert_matrix();
    _distribute(info);
}

void GlobalDyTopoEffectManager::Impl::_assemble(ComputeDyTopoEffectInfo& info)
{
    Timer timer{"Assemble Dytopo Effect"};

    auto vertex_count = global_vertex_manager->positions().size();

    auto reporter_gradient_counts = reporter_gradient_offsets_counts.counts();
    auto reporter_hessian_counts  = reporter_hessian_offsets_counts.counts();
    bool gradient_only            = info.m_gradient_only;

    logger::info("DyTopo Effect Assembly: GradientOnly={}, ComponentFlags={}",
                 info.m_gradient_only,
                 enum_flags_name(info.m_component_flags));

    {
        Timer timer{"Report Extent"};
        for(auto&& [i, reporter] : enumerate(dytopo_effect_reporters.view()))
        {
            reporter_gradient_counts[i] = 0;
            reporter_hessian_counts[i]  = 0;

            if(!has_flags(info.m_component_flags, reporter->component_flags()))
                continue;

            GradientHessianExtentInfo extent_info;
            extent_info.m_gradient_only = gradient_only;
            reporter->report_gradient_hessian_extent(extent_info);

            reporter_gradient_counts[i] = extent_info.m_gradient_count;
            reporter_hessian_counts[i] = gradient_only ? 0 : extent_info.m_hessian_count;
            logger::info("<{}> DyTopo Grad3 count: {}, DyTopo Hess3x3 count: {}",
                         reporter->name(),
                         extent_info.m_gradient_count,
                         extent_info.m_hessian_count);
        }
    }

    {
        Timer timer{"Scan and Allocate"};
        // scan
        reporter_gradient_offsets_counts.scan();
        reporter_hessian_offsets_counts.scan();

        auto total_gradient_count = reporter_gradient_offsets_counts.total_count();
        auto total_hessian_count  = reporter_hessian_offsets_counts.total_count();

        // allocate
        loose_resize_entries(collected_dytopo_effect_gradient, total_gradient_count);
        loose_resize_entries(sorted_dytopo_effect_gradient, total_gradient_count);
        loose_resize_entries(collected_dytopo_effect_hessian, total_hessian_count);
        loose_resize_entries(sorted_dytopo_effect_hessian, total_hessian_count);
        collected_dytopo_effect_gradient.reshape(vertex_count);
        collected_dytopo_effect_hessian.reshape(vertex_count, vertex_count);
    }

    // collect
    for(auto&& [i, reporter] : enumerate(dytopo_effect_reporters.view()))
    {
        if(!has_flags(info.m_component_flags, reporter->component_flags()))
            continue;

        auto [g_offset, g_count] = reporter_gradient_offsets_counts[i];
        auto [h_offset, h_count] = reporter_hessian_offsets_counts[i];

        GradientHessianInfo info;
        info.m_gradient_only = gradient_only;

        info.m_gradients =
            collected_dytopo_effect_gradient.view().subview(g_offset, g_count);
        info.m_hessians = collected_dytopo_effect_hessian.view().subview(h_offset, h_count);

        reporter->assemble(info);
    }
}

void GlobalDyTopoEffectManager::Impl::_convert_matrix()
{
    Timer timer{"Convert Dytopo Matrix"};
    use_raw_full_gradient_distribution = false;
    use_raw_full_hessian_distribution = false;

    static const bool matrixfree_contact = []
    {
#if defined(UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE) && UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE
        const char* e = std::getenv("UIPC_GIPC_MATRIXFREE_CONTACT");
        return e && e[0] != '\0' && e[0] != '0';
#else
        return false;
#endif
    }();

    if(matrixfree_contact)
    {
        auto& from = collected_dytopo_effect_hessian;
        auto& to   = sorted_dytopo_effect_hessian;
        to.reshape(from.rows(), from.cols());
        to.resize_triplets(from.triplet_count());
        if(from.triplet_count() > 0)
        {
            to.row_indices().copy_from(from.row_indices());
            to.col_indices().copy_from(from.col_indices());
            to.values().copy_from(from.values());
        }
    }
    else
    {
        matrix_converter.convert(collected_dytopo_effect_hessian, sorted_dytopo_effect_hessian);
    }

    matrix_converter.convert(collected_dytopo_effect_gradient, sorted_dytopo_effect_gradient);
}

bool GlobalDyTopoEffectManager::Impl::_can_distribute_raw_full_gradient()
{
    return false;
}

bool GlobalDyTopoEffectManager::Impl::_can_distribute_raw_full_hessian()
{
    return false;
}

void GlobalDyTopoEffectManager::Impl::_distribute(ComputeDyTopoEffectInfo& info)
{
    Timer timer{"Distribute Dytopo Effect"};

    using namespace muda;

    auto vertex_count = global_vertex_manager->positions().size();

    for(auto&& [i, receiver] : enumerate(dytopo_effect_receivers.view()))
    {
        DyTopoClassifyInfo classify_info;
        receiver->report(classify_info);

        ClassifiedDyTopoEffectInfo classified_info;
        auto& classified_gradients = classified_dytopo_effect_gradients[i];
        classified_gradients.reshape(vertex_count);
        auto& classified_hessians = classified_dytopo_effect_hessians[i];
        classified_hessians.reshape(vertex_count, vertex_count);

        // 1) report gradient
        if(classify_info.is_diag())
        {
            const auto N = sorted_dytopo_effect_gradient.doublet_count();

            // clear the range in device
            gradient_range = Vector2i{0, 0};

            // partition
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    N,
                    [gradient_range = gradient_range.viewer().name("gradient_range"),
                     dytopo_effect_gradient =
                         std::as_const(sorted_dytopo_effect_gradient).viewer().name("dytopo_effect_gradient"),
                     range = classify_info.gradient_i_range()] __device__(int I) mutable
                    {
                        auto in_range = [](int i, const Vector2i& range)
                        { return i >= range.x() && i < range.y(); };

                        auto&& [i, G]      = dytopo_effect_gradient(I);
                        bool this_in_range = in_range(i, range);

                        if(!this_in_range)
                        {
                            return;
                        }

                        bool prev_in_range = false;
                        if(I > 0)
                        {
                            auto&& [prev_i, prev_G] = dytopo_effect_gradient(I - 1);
                            prev_in_range = in_range(prev_i, range);
                        }
                        bool next_in_range = false;
                        if(I < dytopo_effect_gradient.total_doublet_count() - 1)
                        {
                            auto&& [next_i, next_G] = dytopo_effect_gradient(I + 1);
                            next_in_range = in_range(next_i, range);
                        }

                        // if the prev is not in range, then this is the start of the partition
                        if(!prev_in_range)
                        {
                            gradient_range->x() = I;
                        }
                        // if the next is not in range, then this is the end of the partition
                        if(!next_in_range)
                        {
                            gradient_range->y() = I + 1;
                        }
                    });

            Vector2i h_range = gradient_range;  // copy back

            auto count = h_range.y() - h_range.x();

            loose_resize_entries(classified_gradients, count);

            // fill
            if(count > 0)
            {
                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply(count,
                           [dytopo_effect_gradient = std::as_const(sorted_dytopo_effect_gradient)
                                                         .viewer()
                                                         .name("dytopo_effect_gradient"),
                            classified_gradient = classified_gradients.viewer().name("classified_gradient"),
                            range = h_range] __device__(int I) mutable
                           {
                               auto&& [i, G] = dytopo_effect_gradient(range.x() + I);
                               classified_gradient(I).write(i, G);
                           });
            }

            classified_info.m_gradients = classified_gradients.view();
        }

        // 2) report hessian
        if(!info.m_gradient_only && !classify_info.is_empty())
        {
            const auto N = sorted_dytopo_effect_hessian.triplet_count();

            // +1 for calculate the total count
            loose_resize(selected_hessian, N + 1);
            loose_resize(selected_hessian_offsets, N + 1);

            // select
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    N,
                    [selected_hessian = selected_hessian.view(0, N).viewer().name("selected_hessian"),
                     last =
                         VarView<IndexT>{selected_hessian.data() + N}.viewer().name("last"),
                     dytopo_effect_hessian =
                         sorted_dytopo_effect_hessian.cviewer().name("dytopo_effect_hessian"),
                     i_range = classify_info.hessian_i_range(),
                     j_range = classify_info.hessian_j_range()] __device__(int I) mutable
                    {
                        auto&& [i, j, H] = dytopo_effect_hessian(I);

                        auto in_range = [](int i, const Vector2i& range)
                        { return i >= range.x() && i < range.y(); };

                        selected_hessian(I) =
                            in_range(i, i_range) && in_range(j, j_range) ? 1 : 0;

                        // fill the last one as 0, so that we can calculate the total count
                        // during the exclusive scan
                        if(I == 0)
                            last = 0;
                    });

            // scan
            DeviceScan().ExclusiveSum(selected_hessian.data(),
                                      selected_hessian_offsets.data(),
                                      selected_hessian.size());

            IndexT h_total_count = 0;
            VarView<IndexT>{selected_hessian_offsets.data() + N}.copy_to(&h_total_count);

            loose_resize_entries(classified_hessians, h_total_count);

            // fill
            if(h_total_count > 0)
            {
                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply(N,
                           [selected_hessian = selected_hessian.cviewer().name("selected_hessian"),
                            selected_hessian_offsets =
                                selected_hessian_offsets.cviewer().name("selected_hessian_offsets"),
                            dytopo_effect_hessian =
                                sorted_dytopo_effect_hessian.cviewer().name("dytopo_effect_hessian"),
                            classified_hessian = classified_hessians.viewer().name("classified_hessian"),
                            i_range = classify_info.hessian_i_range(),
                            j_range = classify_info.hessian_j_range()] __device__(int I) mutable
                           {
                               if(selected_hessian(I))
                               {
                                   auto&& [i, j, H] = dytopo_effect_hessian(I);
                                   auto offset = selected_hessian_offsets(I);

                                   classified_hessian(offset).write(i, j, H);
                               }
                           });
            }

            classified_info.m_hessians = classified_hessians.view();
        }

        receiver->receive(classified_info);
    }
}

void GlobalDyTopoEffectManager::Impl::loose_resize_entries(
    muda::DeviceTripletMatrix<Float, 3>& m, SizeT size)
{
    if(size > m.triplet_capacity())
    {
        m.reserve_triplets(size * reserve_ratio);
    }
    m.resize_triplets(size);
}

void GlobalDyTopoEffectManager::Impl::loose_resize_entries(
    muda::DeviceDoubletVector<Float, 3>& v, SizeT size)
{
    if(size > v.doublet_capacity())
    {
        v.reserve_doublets(size * reserve_ratio);
    }
    v.resize_doublets(size);
}
}  // namespace uipc::backend::cuda


namespace uipc::backend::cuda
{
void GlobalDyTopoEffectManager::init()
{
    m_impl.init(world());
}

void GlobalDyTopoEffectManager::compute_dytopo_effect(ComputeDyTopoEffectInfo& info)
{
    m_impl.compute_dytopo_effect(info);
}

void GlobalDyTopoEffectManager::compute_dytopo_effect()
{
    ComputeDyTopoEffectInfo info;
    m_impl.compute_dytopo_effect(info);
}

void GlobalDyTopoEffectManager::add_reporter(DyTopoEffectReporter* reporter)
{
    check_state(SimEngineState::BuildSystems, "add_reporter()");
    UIPC_ASSERT(reporter != nullptr, "reporter is nullptr");
    auto flag = reporter->component_flags();
    UIPC_ASSERT(is_valid_flag(flag),
                "reporter component_flags() is not valid single flag, it's {}",
                enum_flags_name(flag));
    m_impl.dytopo_effect_reporters.register_sim_system(*reporter);

    // classify into contact / non-contact
    if(reporter->component_flags() == EnergyComponentFlags::Contact)
    {
        m_impl.contact_reporters.register_sim_system(*reporter);
    }
    else
    {
        m_impl.non_contact_reporters.register_sim_system(*reporter);
    }
}

void GlobalDyTopoEffectManager::add_receiver(DyTopoEffectReceiver* receiver)
{
    check_state(SimEngineState::BuildSystems, "add_receiver()");
    UIPC_ASSERT(receiver != nullptr, "receiver is nullptr");
    m_impl.dytopo_effect_receivers.register_sim_system(*receiver);
}
}  // namespace uipc::backend::cuda
#endif
