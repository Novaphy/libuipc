#include <affine_body/abd_line_search_reporter.h>
#include <affine_body/affine_body_constitution.h>
#include <muda/cub/device/device_reduce.h>
#include <kernel_cout.h>
#include <muda/ext/eigen/log_proxy.h>
#include <affine_body/abd_line_search_subreporter.h>
#include <affine_body/affine_body_kinetic.h>

namespace uipc::backend::cuda
{

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
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
#endif

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

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    checkCudaErrors(cudaMemcpy(abd().body_id_to_q_temp.data(),
                               abd().body_id_to_q.data(),
                               sizeof(Vector12) * abd().body_count(),
                               cudaMemcpyDeviceToDevice));
#else
    BufferLaunch().template copy<Vector12>(abd().body_id_to_q_temp.view(),
                                           abd().body_id_to_q.view());
#endif
}

void ABDLineSearchReporter::Impl::step_forward(LineSearcher::StepInfo& info)
{
    using namespace muda;
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    {
        int n = static_cast<int>(abd().abd_body_count);
        if(n > 0)
        {
            std::vector<IndexT> h_fixed(n);
            std::vector<Float> h_qt(n*12), h_dq(n*12);
            cudaMemcpy(h_fixed.data(), abd().body_id_to_is_fixed.data(), n*sizeof(IndexT), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_qt.data(), abd().body_id_to_q_temp.data(), n*12*sizeof(Float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_dq.data(), abd().body_id_to_dq.data(), n*12*sizeof(Float), cudaMemcpyDeviceToHost);

            std::vector<Float> h_q(h_qt);
            for(int i = 0; i < n; ++i)
            {
                if(h_fixed[i]) continue;
                for(int k = 0; k < 12; ++k)
                    h_q[i*12+k] = h_qt[i*12+k] + info.alpha * h_dq[i*12+k];
            }

            cudaMemcpy((void*)abd().body_id_to_q.data(), h_q.data(), n*12*sizeof(Float), cudaMemcpyHostToDevice);
        }
    }
#else
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
#endif
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

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        // Skip zeroing kernel on CoreX: bdf1_kinetic_energy_kernel already handles is_fixed/ext_kinetic
#else
        using namespace muda;

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
#endif

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        {
            int nk = static_cast<int>(body_id_to_kinetic_energy.size());
            std::vector<Float> hke(nk);
            checkCudaErrors(cudaMemcpy(hke.data(), body_id_to_kinetic_energy.data(),
                                       sizeof(Float) * nk, cudaMemcpyDeviceToHost));
            Float sum = 0;
            for(int i = 0; i < nk; ++i) sum += hke[i];
            checkCudaErrors(cudaMemcpy(abd_kinetic_energy.data(), &sum, sizeof(Float), cudaMemcpyHostToDevice));
        }
#else
        DeviceReduce().Sum(body_id_to_kinetic_energy.data(),
                           abd_kinetic_energy.data(),
                           body_id_to_kinetic_energy.size());
#endif
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

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        {
            int ns = static_cast<int>(body_id_to_shape_energy.size());
            std::vector<Float> hse(ns);
            checkCudaErrors(cudaMemcpy(hse.data(), body_id_to_shape_energy.data(),
                                       sizeof(Float) * ns, cudaMemcpyDeviceToHost));
            Float sum = 0;
            for(int i = 0; i < ns; ++i) sum += hse[i];
            checkCudaErrors(cudaMemcpy(abd_shape_energy.data(), &sum, sizeof(Float), cudaMemcpyHostToDevice));
        }
#else
        DeviceReduce().Sum(body_id_to_shape_energy.data(),
                           abd_shape_energy.data(),
                           body_id_to_shape_energy.size());
#endif
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
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        {
            int nr = static_cast<int>(reporter_energies.size());
            Float sum = 0;
            if(nr > 0)
            {
                std::vector<Float> hre(nr);
                checkCudaErrors(cudaMemcpy(hre.data(), reporter_energies.data(),
                                           sizeof(Float) * nr, cudaMemcpyDeviceToHost));
                for(int i = 0; i < nr; ++i) sum += hre[i];
            }
            checkCudaErrors(cudaMemcpy(total_reporter_energy.data(), &sum, sizeof(Float), cudaMemcpyHostToDevice));
        }
#else
        DeviceReduce().Sum(reporter_energies.data(),
                           total_reporter_energy.data(),
                           reporter_energies.size());
#endif
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
