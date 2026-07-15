#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#include <affine_body/abd_line_search_reporter.h>
#include <affine_body/affine_body_constitution.h>
#include <kernel_cout.h>
#include <muda/ext/eigen/log_proxy.h>
#include <affine_body/abd_line_search_subreporter.h>
#include <affine_body/affine_body_kinetic.h>
#include <muda/check/check_cuda_errors.h>
#include <algorithm>
#include <cstdlib>

namespace uipc::backend::cuda
{
namespace
{
struct CorexLineSearchFloatReadback
{
    Float*       pinned = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t  ready = nullptr;

    CorexLineSearchFloatReadback()
    {
        Float* tmp = nullptr;
        if(cudaMallocHost(reinterpret_cast<void**>(&tmp), sizeof(Float)) == cudaSuccess)
            pinned = tmp;
        checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        checkCudaErrors(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    }
};

inline Float corex_line_search_readback_float(const Float* value)
{
    static CorexLineSearchFloatReadback readback;
    if(!readback.pinned)
    {
        Float host_value = 0.0;
        checkCudaErrors(cudaMemcpy(&host_value, value, sizeof(Float), cudaMemcpyDeviceToHost));
        return host_value;
    }

    checkCudaErrors(cudaEventRecord(readback.ready, 0));
    checkCudaErrors(cudaStreamWaitEvent(readback.stream, readback.ready, 0));
    checkCudaErrors(cudaMemcpyAsync(readback.pinned,
                                    value,
                                    sizeof(Float),
                                    cudaMemcpyDeviceToHost,
                                    readback.stream));
    checkCudaErrors(cudaStreamSynchronize(readback.stream));
    return *readback.pinned;
}
}  // namespace

__global__ void kernel_step_forward(int            n,
                                    Float          alpha,
                                    const IndexT*  is_fixed,
                                    const Float*   q_temps,
                                    Float*         qs,
                                    const Float*   dqs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    if(is_fixed[i]) return;
    for(int k = 0; k < 12; ++k)
        qs[i * 12 + k] = q_temps[i * 12 + k] + alpha * dqs[i * 12 + k];
}

__global__ void kernel_sum_line_search_energy(int           body_count,
                                              int           reporter_count,
                                              const IndexT* is_fixed,
                                              const IndexT* external_kinetic,
                                              const Float*  kinetic_energy,
                                              const Float*  shape_energy,
                                              const Float*  reporter_energy,
                                              Float*        total_energy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Float local = 0.0;

    if(i < body_count)
    {
        if(!is_fixed[i] && !external_kinetic[i])
            local += kinetic_energy[i];
        local += shape_energy[i];
    }

    if(i < reporter_count)
        local += reporter_energy[i];

    __shared__ Float block_sum[256];
    block_sum[threadIdx.x] = local;
    __syncthreads();

    for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if(threadIdx.x < stride)
            block_sum[threadIdx.x] += block_sum[threadIdx.x + stride];
        __syncthreads();
    }

    if(threadIdx.x == 0 && block_sum[0] != 0.0)
        atomicAdd(total_energy, block_sum[0]);
}

REGISTER_SIM_SYSTEM(ABDLineSearchReporter);

void ABDLineSearchReporter::do_build(LineSearchReporter::BuildInfo& info)
{
    m_impl.affine_body_dynamics = require<AffineBodyDynamics>();
}

void ABDLineSearchReporter::Impl::init(LineSearchReporter::InitInfo& info)
{
    auto reporter_view = reporters.view();
    for(auto&& [i, R] : enumerate(reporter_view))
        R->m_index = i;  // Assign index for each reporter
    for(auto&& [i, R] : enumerate(reporter_view))
        R->init();

    reporter_energy_offsets_counts.resize(reporter_view.size());

    // Single-scalar device buffers (was muda::DeviceVar): avoid cudaMalloc in SimSystem ctor;
    // Iluvatar/Corex has been observed to block there during build_systems().
    abd_kinetic_energy.resize(1);
    abd_shape_energy.resize(1);
    total_reporter_energy.resize(1);
}

void ABDLineSearchReporter::Impl::record_start_point(LineSearcher::RecordInfo& info)
{
    using namespace muda;

    checkCudaErrors(cudaMemcpyAsync(abd().body_id_to_q_temp.data(),
                                    abd().body_id_to_q.data(),
                                    sizeof(Vector12) * abd().body_count(),
                                    cudaMemcpyDeviceToDevice));
}

void ABDLineSearchReporter::Impl::step_forward(LineSearcher::StepInfo& info)
{
    using namespace muda;
    int n = static_cast<int>(abd().abd_body_count);
    if(n <= 0)
        return;

    constexpr int block = 256;
    int           grid  = (n + block - 1) / block;
    kernel_step_forward<<<grid, block>>>(n,
                                         info.alpha,
                                         abd().body_id_to_is_fixed.data(),
                                         reinterpret_cast<const Float*>(abd().body_id_to_q_temp.data()),
                                         reinterpret_cast<Float*>(abd().body_id_to_q.data()),
                                         reinterpret_cast<const Float*>(abd().body_id_to_dq.data()));
    checkCudaErrors(cudaGetLastError());
}

void ABDLineSearchReporter::Impl::compute_energy(LineSearcher::ComputeEnergyInfo& info)
{
    using namespace muda;

    auto body_count = abd().body_count();

    // Compute kinetic energy
    {
        body_id_to_kinetic_energy.resize(body_count);

        ABDLineSearchReporter::ComputeEnergyInfo this_info;
        this_info.m_energies = body_id_to_kinetic_energy.view();
        this_info.m_dt       = info.dt();

        abd().kinetic->compute_energy(this_info);
    }

    // Compute shape energy
    {
        body_id_to_shape_energy.resize(body_count);

        for(auto&& [i, cst] : enumerate(abd().constitutions.view()))
        {
            auto shape_energy = abd().subview(body_id_to_shape_energy, cst->m_index);

            ABDLineSearchReporter::ComputeEnergyInfo this_info;
            this_info.m_energies = shape_energy;
            this_info.m_dt       = info.dt();
            cst->compute_energy(this_info);
        }

    }

    // Collect the energy from other reporters
    {
        auto         reporter_view = reporters.view();
        span<IndexT> counts        = reporter_energy_offsets_counts.counts();
        for(auto&& [i, R] : enumerate(reporter_view))
        {
            ReportExtentInfo this_info;
            R->report_extent(this_info);
            counts[i] = this_info.m_energy_count;
        }

        reporter_energy_offsets_counts.scan();
        reporter_energies.resize(reporter_energy_offsets_counts.total_count());

        for(auto&& [i, R] : enumerate(reporter_view))
        {
            ComputeEnergyInfo this_info;
            auto [offset, count] = reporter_energy_offsets_counts[i];
            this_info.m_energies = reporter_energies.view(offset, count);
            this_info.m_dt       = info.dt();
            R->compute_energy(this_info);
        }

    }

    const int n_body     = static_cast<int>(body_count);
    const int n_reporter = static_cast<int>(reporter_energies.size());
    const int n_sum      = std::max(n_body, n_reporter);

    checkCudaErrors(cudaMemsetAsync(total_reporter_energy.data(), 0, sizeof(Float)));
    if(n_sum > 0)
    {
        constexpr int block = 256;
        int           grid  = (n_sum + block - 1) / block;
        kernel_sum_line_search_energy<<<grid, block>>>(n_body,
                                                       n_reporter,
                                                       abd().body_id_to_is_fixed.data(),
                                                       abd().body_id_to_external_kinetic.data(),
                                                       body_id_to_kinetic_energy.data(),
                                                       body_id_to_shape_energy.data(),
                                                       reporter_energies.data(),
                                                       total_reporter_energy.data());
        checkCudaErrors(cudaGetLastError());
    }

    Float E = corex_line_search_readback_float(total_reporter_energy.data());

    static const bool trace_linear_system =
        std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM") != nullptr;
    if(trace_linear_system)
        logger::info("[corex_trace][energy] total={}", E);

    info.energy(E);
}

void ABDLineSearchReporter::do_init(LineSearchReporter::InitInfo& info)
{
    m_impl.init(info);
}

void ABDLineSearchReporter::do_record_start_point(LineSearcher::RecordInfo& info)
{
    m_impl.record_start_point(info);
}

void ABDLineSearchReporter::do_step_forward(LineSearcher::StepInfo& info)
{
    m_impl.step_forward(info);
}

void ABDLineSearchReporter::do_compute_energy(LineSearcher::ComputeEnergyInfo& info)
{
    m_impl.compute_energy(info);
}

void ABDLineSearchReporter::add_reporter(ABDLineSearchSubreporter* reporter)
{
    UIPC_ASSERT(reporter, "reporter is null");
    check_state(SimEngineState::BuildSystems, "add_reporter()");
    m_impl.reporters.register_sim_system(*reporter);
}
}  // namespace uipc::backend::cuda
#else
#include <affine_body/abd_line_search_reporter.h>
#include <affine_body/affine_body_constitution.h>
#include <muda/cub/device/device_reduce.h>
#include <kernel_cout.h>
#include <muda/ext/eigen/log_proxy.h>
#include <affine_body/abd_line_search_subreporter.h>
#include <affine_body/affine_body_kinetic.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(ABDLineSearchReporter);

static __global__ void kernel_sum_abd_line_search_energy(const Float* kinetic,
                                                         const Float* shape,
                                                         const Float* other,
                                                         Float*       total)
{
    total[0] = kinetic[0] + shape[0] + other[0];
}

void ABDLineSearchReporter::do_build(LineSearchReporter::BuildInfo& info)
{
    m_impl.affine_body_dynamics = require<AffineBodyDynamics>();
}

void ABDLineSearchReporter::Impl::init(LineSearchReporter::InitInfo& info)
{
    auto reporter_view = reporters.view();
    for(auto&& [i, R] : enumerate(reporter_view))
        R->m_index = i;  // Assign index for each reporter
    for(auto&& [i, R] : enumerate(reporter_view))
        R->init();

    reporter_energy_offsets_counts.resize(reporter_view.size());
}

void ABDLineSearchReporter::Impl::record_start_point(LineSearcher::RecordInfo& info)
{
    using namespace muda;

    BufferLaunch().template copy<Vector12>(abd().body_id_to_q_temp.view(),
                                           abd().body_id_to_q.view());
}

void ABDLineSearchReporter::Impl::step_forward(LineSearcher::StepInfo& info)
{
    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(abd().abd_body_count,
               [is_fixed = abd().body_id_to_is_fixed.cviewer().name("is_fixed"),
                q_temps  = abd().body_id_to_q_temp.cviewer().name("q_temps"),
                qs       = abd().body_id_to_q.viewer().name("qs"),
                dqs      = abd().body_id_to_dq.cviewer().name("dqs"),
                alpha    = info.alpha] __device__(int i) mutable
               {
                   if(is_fixed(i))
                       return;
                   qs(i) = q_temps(i) + alpha * dqs(i);
               });
}

void ABDLineSearchReporter::Impl::compute_energy(LineSearcher::ComputeEnergyInfo& info)
{
    using namespace muda;

    auto body_count = abd().body_count();

    // Compute kinetic energy
    {
        body_id_to_kinetic_energy.resize(body_count);

        ABDLineSearchReporter::ComputeEnergyInfo this_info;
        this_info.m_energies = body_id_to_kinetic_energy.view();
        this_info.m_dt       = info.dt();

        abd().kinetic->compute_energy(this_info);

        using namespace muda;

        // Zero out the kinetic energy of fixed bodies and bodies with external kinetic
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(abd().abd_body_count,
                   [is_fixed = abd().body_id_to_is_fixed.cviewer().name("is_fixed"),
                    external_kinetic =
                        abd().body_id_to_external_kinetic.cviewer().name("external_kinetic"),
                    kinetic_energy = body_id_to_kinetic_energy.viewer().name(
                        "kinetic_energy")] __device__(int i) mutable
                   {
                       if(is_fixed(i) || external_kinetic(i))
                       {
                           kinetic_energy(i) = 0.0;
                       }
                   });

        // Sum up the kinetic energy
        DeviceReduce().Sum(body_id_to_kinetic_energy.data(),
                           abd_kinetic_energy.data(),
                           body_id_to_kinetic_energy.size());
    }

    // Compute shape energy
    {
        body_id_to_shape_energy.resize(body_count);

        // Distribute the computation of shape energy to each constitution
        for(auto&& [i, cst] : enumerate(abd().constitutions.view()))
        {
            auto shape_energy = abd().subview(body_id_to_shape_energy, cst->m_index);

            ABDLineSearchReporter::ComputeEnergyInfo this_info;
            this_info.m_energies = shape_energy;
            this_info.m_dt       = info.dt();
            cst->compute_energy(this_info);
        }

        // Sum up the shape energy
        DeviceReduce().Sum(body_id_to_shape_energy.data(),
                           abd_shape_energy.data(),
                           body_id_to_shape_energy.size());
    }

    // Collect the energy from other reporters
    {
        auto         reporter_view = reporters.view();
        span<IndexT> counts        = reporter_energy_offsets_counts.counts();
        for(auto&& [i, R] : enumerate(reporter_view))
        {
            ReportExtentInfo this_info;
            R->report_extent(this_info);
            counts[i] = this_info.m_energy_count;
        }

        reporter_energy_offsets_counts.scan();
        reporter_energies.resize(reporter_energy_offsets_counts.total_count());

        for(auto&& [i, R] : enumerate(reporter_view))
        {
            ComputeEnergyInfo this_info;
            auto [offset, count] = reporter_energy_offsets_counts[i];
            this_info.m_energies = reporter_energies.view(offset, count);
            this_info.m_dt       = info.dt();
            R->compute_energy(this_info);
        }

        // Compute the total energy from all reporters
        DeviceReduce().Sum(reporter_energies.data(),
                           total_reporter_energy.data(),
                           reporter_energies.size());
    }

    kernel_sum_abd_line_search_energy<<<1, 1>>>(abd_kinetic_energy.data(),
                                                abd_shape_energy.data(),
                                                total_reporter_energy.data(),
                                                total_reporter_energy.data());
    checkCudaErrors(cudaGetLastError());

    Float E = total_reporter_energy;
    info.energy(E);
}

void ABDLineSearchReporter::do_init(LineSearchReporter::InitInfo& info)
{
    m_impl.init(info);
}

void ABDLineSearchReporter::do_record_start_point(LineSearcher::RecordInfo& info)
{
    m_impl.record_start_point(info);
}

void ABDLineSearchReporter::do_step_forward(LineSearcher::StepInfo& info)
{
    m_impl.step_forward(info);
}

void ABDLineSearchReporter::do_compute_energy(LineSearcher::ComputeEnergyInfo& info)
{
    m_impl.compute_energy(info);
}

void ABDLineSearchReporter::add_reporter(ABDLineSearchSubreporter* reporter)
{
    UIPC_ASSERT(reporter, "reporter is null");
    check_state(SimEngineState::BuildSystems, "add_reporter()");
    m_impl.reporters.register_sim_system(*reporter);
}
}  // namespace uipc::backend::cuda
#endif
