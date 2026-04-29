#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#include <affine_body/abd_line_search_reporter.h>
#include <affine_body/affine_body_constitution.h>
#include <muda/cub/device/device_reduce.h>
#include <kernel_cout.h>
#include <muda/ext/eigen/log_proxy.h>
#include <affine_body/abd_line_search_subreporter.h>
#include <affine_body/affine_body_kinetic.h>
#include <muda/check/check_cuda_errors.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace uipc::backend::cuda
{

namespace
{
__device__ Float corex_abd_mass_mul_dot(const ABDJacobiDyadicMass& mass,
                                        const Vector12&            dq)
{
    const Float      m = mass.mass();
    const Vector3&   x = mass.mass_times_x_bar();
    const Matrix3x3& D = mass.mass_times_dyadic_x_bar();

    Float ret = 0;
    Float mdq0 = x[0] * dq[3] + x[1] * dq[4] + x[2] * dq[5] + m * dq[0];
    Float mdq1 = x[0] * dq[6] + x[1] * dq[7] + x[2] * dq[8] + m * dq[1];
    Float mdq2 = x[0] * dq[9] + x[1] * dq[10] + x[2] * dq[11] + m * dq[2];
    ret += dq[0] * mdq0 + dq[1] * mdq1 + dq[2] * mdq2;

    for(int r = 0; r < 3; ++r)
    {
        Float mdq3 = D(r, 0) * dq[3] + D(r, 1) * dq[4] + D(r, 2) * dq[5] + x[r] * dq[0];
        Float mdq6 = D(r, 0) * dq[6] + D(r, 1) * dq[7] + D(r, 2) * dq[8] + x[r] * dq[1];
        Float mdq9 = D(r, 0) * dq[9] + D(r, 1) * dq[10] + D(r, 2) * dq[11] + x[r] * dq[2];
        ret += dq[3 + r] * mdq3 + dq[6 + r] * mdq6 + dq[9 + r] * mdq9;
    }
    return ret;
}

__global__ void kernel_reduce_sum_float(int n, const Float* values, Float* out)
{
    __shared__ Float block_sum[256];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    Float local = 0;
    while(i < n)
    {
        local += values[i];
        i += blockDim.x * gridDim.x;
    }
    block_sum[tid] = local;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(tid < stride)
            block_sum[tid] += block_sum[tid + stride];
        __syncthreads();
    }
    if(tid == 0)
        atomicAdd(out, block_sum[0]);
}

__global__ void kernel_abd_fused_kinetic_energy_sum(
    int                                  n,
    const Vector12* __restrict__         qs,
    const Vector12* __restrict__         q_tildes,
    const ABDJacobiDyadicMass* __restrict__ masses,
    const IndexT* __restrict__           is_fixed,
    const IndexT* __restrict__           external_kinetic,
    Float* __restrict__                  out)
{
    __shared__ Float block_sum[256];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    Float local = 0;
    while(i < n)
    {
        if(!is_fixed[i] && !external_kinetic[i])
        {
            Vector12 dq = qs[i] - q_tildes[i];
            local += Float(0.5) * corex_abd_mass_mul_dot(masses[i], dq);
        }
        i += blockDim.x * gridDim.x;
    }

    block_sum[tid] = local;
    __syncthreads();
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(tid < stride)
            block_sum[tid] += block_sum[tid + stride];
        __syncthreads();
    }
    if(tid == 0)
        atomicAdd(out, block_sum[0]);
}

void corex_reduce_sum_float(int n, const Float* values, Float* out)
{
    checkCudaErrors(cudaMemset(out, 0, sizeof(Float)));
    if(n <= 0)
        return;
    constexpr int block = 256;
    int grid = std::min((n + block - 1) / block, 128);
    kernel_reduce_sum_float<<<grid, block>>>(n, values, out);
    checkCudaErrors(cudaGetLastError());
}

void corex_log_energy_ab(const char* kind, Float host, Float gpu)
{
    Float abs_err = std::abs(host - gpu);
    Float denom   = std::max(std::abs(host), Float(1e-12));
    Float rel_err = abs_err / denom;
    std::fprintf(stderr,
                 "[corex_abd_energy_ab] kind=%s host=%.9g gpu=%.9g abs=%.9g rel=%.9g\n",
                 kind,
                 static_cast<float>(host),
                 static_cast<float>(gpu),
                 static_cast<float>(abs_err),
                 static_cast<float>(rel_err));
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

    checkCudaErrors(cudaMemcpy(abd().body_id_to_q_temp.data(),
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

    if(std::getenv("UIPC_COREX_ABD_LINE_SEARCH_HOST_STEP"))
    {
        std::vector<IndexT> h_fixed(n);
        std::vector<Float>  h_qt(n * 12), h_dq(n * 12);
        checkCudaErrors(cudaMemcpy(h_fixed.data(), abd().body_id_to_is_fixed.data(), n * sizeof(IndexT), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_qt.data(), abd().body_id_to_q_temp.data(), n * 12 * sizeof(Float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_dq.data(), abd().body_id_to_dq.data(), n * 12 * sizeof(Float), cudaMemcpyDeviceToHost));

        std::vector<Float> h_q(h_qt);
        for(int i = 0; i < n; ++i)
        {
            if(h_fixed[i])
                continue;
            for(int k = 0; k < 12; ++k)
                h_q[i * 12 + k] = h_qt[i * 12 + k] + info.alpha * h_dq[i * 12 + k];
        }

        checkCudaErrors(cudaMemcpy((void*)abd().body_id_to_q.data(), h_q.data(), n * 12 * sizeof(Float), cudaMemcpyHostToDevice));
        return;
    }

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
        const int nk = static_cast<int>(body_count);
        const bool fused_kinetic = std::getenv("UIPC_COREX_ABD_FUSED_ENERGY_GPU") != nullptr;
        if(fused_kinetic)
        {
            checkCudaErrors(cudaMemset(abd_kinetic_energy.data(), 0, sizeof(Float)));
            if(nk > 0)
            {
                constexpr int block = 256;
                int           grid  = std::min((nk + block - 1) / block, 128);
                kernel_abd_fused_kinetic_energy_sum<<<grid, block>>>(
                    nk,
                    abd().body_id_to_q.data(),
                    abd().body_id_to_q_tilde.data(),
                    abd().body_id_to_abd_mass.data(),
                    abd().body_id_to_is_fixed.data(),
                    abd().body_id_to_external_kinetic.data(),
                    abd_kinetic_energy.data());
                checkCudaErrors(cudaGetLastError());
            }

            if(std::getenv("UIPC_COREX_ABD_ENERGY_AB_COMPARE"))
            {
                body_id_to_kinetic_energy.resize(body_count);
                ABDLineSearchReporter::ComputeEnergyInfo this_info;
                this_info.m_energies = body_id_to_kinetic_energy.view();
                this_info.m_dt       = info.dt();
                abd().kinetic->compute_energy(this_info);

                std::vector<Float> hke(nk);
                checkCudaErrors(cudaMemcpy(hke.data(),
                                           body_id_to_kinetic_energy.data(),
                                           sizeof(Float) * nk,
                                           cudaMemcpyDeviceToHost));
                Float host_sum = 0;
                for(int i = 0; i < nk; ++i)
                    host_sum += hke[i];
                Float gpu_sum = 0;
                checkCudaErrors(cudaMemcpy(&gpu_sum,
                                           abd_kinetic_energy.data(),
                                           sizeof(Float),
                                           cudaMemcpyDeviceToHost));
                corex_log_energy_ab("kinetic", host_sum, gpu_sum);
            }
        }
        else
        {
            body_id_to_kinetic_energy.resize(body_count);

            ABDLineSearchReporter::ComputeEnergyInfo this_info;
            this_info.m_energies = body_id_to_kinetic_energy.view();
            this_info.m_dt       = info.dt();

            abd().kinetic->compute_energy(this_info);

            if(std::getenv("UIPC_COREX_ABD_ENERGY_HOST_SUM")
               || (std::getenv("UIPC_COREX_ABD_ENERGY_REDUCTION_GPU") == nullptr
                   && std::getenv("UIPC_COREX_ABD_LIGHT_REDUCTION_GPU") == nullptr))
            {
                std::vector<Float> hke(nk);
                checkCudaErrors(cudaMemcpy(hke.data(), body_id_to_kinetic_energy.data(), sizeof(Float) * nk, cudaMemcpyDeviceToHost));
                Float sum = 0;
                for(int i = 0; i < nk; ++i)
                    sum += hke[i];
                checkCudaErrors(cudaMemcpy(abd_kinetic_energy.data(), &sum, sizeof(Float), cudaMemcpyHostToDevice));
            }
            else if(std::getenv("UIPC_COREX_ABD_LIGHT_REDUCTION_GPU"))
            {
                corex_reduce_sum_float(nk, body_id_to_kinetic_energy.data(), abd_kinetic_energy.data());
            }
            else
            {
                DeviceReduce().Sum(body_id_to_kinetic_energy.data(),
                                   abd_kinetic_energy.data(),
                                   body_id_to_kinetic_energy.size());
                checkCudaErrors(cudaGetLastError());
            }
        }
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

        if(std::getenv("UIPC_COREX_ABD_ENERGY_HOST_SUM")
           || (std::getenv("UIPC_COREX_ABD_ENERGY_REDUCTION_GPU") == nullptr
               && std::getenv("UIPC_COREX_ABD_LIGHT_REDUCTION_GPU") == nullptr))
        {
            int ns = static_cast<int>(body_id_to_shape_energy.size());
            std::vector<Float> hse(ns);
            checkCudaErrors(cudaMemcpy(hse.data(), body_id_to_shape_energy.data(),
                                       sizeof(Float) * ns, cudaMemcpyDeviceToHost));
            Float sum = 0;
            for(int i = 0; i < ns; ++i)
                sum += hse[i];
            checkCudaErrors(cudaMemcpy(abd_shape_energy.data(), &sum, sizeof(Float), cudaMemcpyHostToDevice));
        }
        else if(std::getenv("UIPC_COREX_ABD_LIGHT_REDUCTION_GPU"))
        {
            int ns = static_cast<int>(body_id_to_shape_energy.size());
            corex_reduce_sum_float(ns, body_id_to_shape_energy.data(), abd_shape_energy.data());

            if(std::getenv("UIPC_COREX_ABD_ENERGY_AB_COMPARE"))
            {
                std::vector<Float> hse(ns);
                checkCudaErrors(cudaMemcpy(hse.data(),
                                           body_id_to_shape_energy.data(),
                                           sizeof(Float) * ns,
                                           cudaMemcpyDeviceToHost));
                Float host_sum = 0;
                for(int i = 0; i < ns; ++i)
                    host_sum += hse[i];
                Float gpu_sum = 0;
                checkCudaErrors(cudaMemcpy(&gpu_sum,
                                           abd_shape_energy.data(),
                                           sizeof(Float),
                                           cudaMemcpyDeviceToHost));
                corex_log_energy_ab("shape", host_sum, gpu_sum);
            }
        }
        else
        {
            DeviceReduce().Sum(body_id_to_shape_energy.data(),
                               abd_shape_energy.data(),
                               body_id_to_shape_energy.size());
            checkCudaErrors(cudaGetLastError());
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

        // Compute the total energy from all reporters
        if(std::getenv("UIPC_COREX_ABD_ENERGY_HOST_SUM")
           || (std::getenv("UIPC_COREX_ABD_ENERGY_REDUCTION_GPU") == nullptr
               && std::getenv("UIPC_COREX_ABD_LIGHT_REDUCTION_GPU") == nullptr))
        {
            int nr = static_cast<int>(reporter_energies.size());
            Float sum = 0;
            if(nr > 0)
            {
                std::vector<Float> hre(nr);
                checkCudaErrors(cudaMemcpy(hre.data(), reporter_energies.data(),
                                           sizeof(Float) * nr, cudaMemcpyDeviceToHost));
                for(int i = 0; i < nr; ++i)
                    sum += hre[i];
            }
            checkCudaErrors(cudaMemcpy(total_reporter_energy.data(), &sum, sizeof(Float), cudaMemcpyHostToDevice));
        }
        else if(std::getenv("UIPC_COREX_ABD_LIGHT_REDUCTION_GPU"))
        {
            int nr = static_cast<int>(reporter_energies.size());
            corex_reduce_sum_float(nr, reporter_energies.data(), total_reporter_energy.data());

            if(std::getenv("UIPC_COREX_ABD_ENERGY_AB_COMPARE"))
            {
                Float host_sum = 0;
                if(nr > 0)
                {
                    std::vector<Float> hre(nr);
                    checkCudaErrors(cudaMemcpy(hre.data(),
                                               reporter_energies.data(),
                                               sizeof(Float) * nr,
                                               cudaMemcpyDeviceToHost));
                    for(int i = 0; i < nr; ++i)
                        host_sum += hre[i];
                }
                Float gpu_sum = 0;
                checkCudaErrors(cudaMemcpy(&gpu_sum,
                                           total_reporter_energy.data(),
                                           sizeof(Float),
                                           cudaMemcpyDeviceToHost));
                corex_log_energy_ab("reporter", host_sum, gpu_sum);
            }
        }
        else
        {
            int nr = static_cast<int>(reporter_energies.size());
            if(nr > 0)
            {
                DeviceReduce().Sum(reporter_energies.data(),
                                   total_reporter_energy.data(),
                                   nr);
                checkCudaErrors(cudaGetLastError());
            }
            else
            {
                Float z = 0.f;
                checkCudaErrors(cudaMemcpy(total_reporter_energy.data(), &z, sizeof(Float), cudaMemcpyHostToDevice));
            }
        }
    }

    // Copy from device to host
    Float K, shape_E, other_E;
    abd_kinetic_energy.view().copy_to(&K);
    abd_shape_energy.view().copy_to(&shape_E);
    total_reporter_energy.view().copy_to(&other_E);

    Float E = K + shape_E + other_E;

    if(std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM"))
        logger::info("[corex_trace][energy] K={}, shape={}, other={}, total={}", K, shape_E, other_E, E);

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

    // Copy from device to host
    Float K       = abd_kinetic_energy;
    Float shape_E = abd_shape_energy;
    Float other_E = total_reporter_energy;

    Float E = K + shape_E + other_E;

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
