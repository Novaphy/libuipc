#include <affine_body/abd_linear_subsystem.h>
#include <sim_engine.h>
#include <kernel_cout.h>
#include <muda/ext/eigen.h>
#include <utils/matrix_assembler.h>
#include <utils/matrix_unpacker.h>
#include <uipc/builtin/attribute_name.h>
#include <affine_body/inter_affine_body_constitution_manager.h>
#include <affine_body/abd_linear_subsystem_reporter.h>
#include <affine_body/affine_body_kinetic.h>
#include <affine_body/affine_body_constitution.h>
#include <utils/report_extent_check.h>
#include <cstdlib>
#include <vector>

namespace uipc::backend::cuda
{
UIPC_HOST UIPC_DEVICE void zero_out_lower(Matrix12x12& H)
{
    // DEBUG: Some CUDA backends may populate only one triangle.
    // Clearing the "lower" triangle can therefore wipe out all values.
    // Keep this as a no-op to validate assembly is producing non-zero blocks.
}
}  // namespace uipc::backend::cuda

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(ABDLinearSubsystem);

// ref: https://github.com/spiriMirror/libuipc/issues/271
constexpr U64 ABDLinearSubsystemUID = 0ull;

void ABDLinearSubsystem::do_build(DiagLinearSubsystem::BuildInfo& info)
{
    m_impl.affine_body_dynamics        = require<AffineBodyDynamics>();
    m_impl.affine_body_vertex_reporter = require<AffineBodyVertexReporter>();
    auto attr = world().scene().config().find<Float>("dt");
    m_impl.dt = attr->view()[0];

    m_impl.dytopo_effect_receiver = find<ABDDyTopoEffectReceiver>();
}

void ABDLinearSubsystem::Impl::init()
{
    auto reporter_view = reporters.view();
    for(auto&& [i, r] : enumerate(reporter_view))
        r->m_index = i;
    for(auto& r : reporter_view)
        r->init();

    reporter_gradient_offsets_counts.resize(reporter_view.size());
    reporter_hessian_offsets_counts.resize(reporter_view.size());

    SizeT body_count = abd().body_count();
    body_id_to_shape_hessian.resize(body_count);
    body_id_to_shape_gradient.resize(body_count);
    body_id_to_kinetic_hessian.resize(body_count);
    body_id_to_kinetic_gradient.resize(body_count);
    diag_hessian.resize(body_count);
}

void ABDLinearSubsystem::Impl::report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info)
{
    info.extent(abd().body_count() * 12);
}

void ABDLinearSubsystem::Impl::receive_init_dof_info(WorldVisitor& w,
                                                     GlobalLinearSystem::InitDofInfo& info)
{
    auto& geo_infos = abd().geo_infos;
    auto  geo_slots = w.scene().geometries();

    IndexT offset = info.dof_offset();

    // fill the dof_offset and dof_count for each geometry
    affine_body_dynamics->for_each(
        geo_slots,
        [&](const AffineBodyDynamics::ForEachInfo& foreach_info, geometry::SimplicialComplex& sc)
        {
            auto I          = foreach_info.global_index();
            auto dof_offset = sc.meta().find<IndexT>(builtin::dof_offset);
            UIPC_ASSERT(dof_offset, "dof_offset not found on ABD mesh why can it happen?");
            auto dof_count = sc.meta().find<IndexT>(builtin::dof_count);
            UIPC_ASSERT(dof_count, "dof_count not found on ABD mesh why can it happen?");

            IndexT this_dof_count = 12 * sc.instances().size();
            view(*dof_offset)[0]  = offset;
            view(*dof_count)[0]   = this_dof_count;

            offset += this_dof_count;
        });

    UIPC_ASSERT(offset == info.dof_offset() + info.dof_count(), "dof size mismatch");
}

void ABDLinearSubsystem::Impl::report_extent(GlobalLinearSystem::DiagExtentInfo& info)
{
    // 1. Gradient Count
    constexpr SizeT G12_to_dof = 12;
    SizeT           body_count = abd().body_count();
    auto            dof_count  = body_count * G12_to_dof;

    auto has_complement =
        has_flags(info.component_flags(), GlobalLinearSystem::ComponentFlags::Complement);

    SizeT H12x12_count = 0;

    if(has_complement)
    {
        // 1) Body hessian: kinetic + shape
        if(!info.gradient_only())
            H12x12_count += abd().body_count();

        // 2) Reporters
        auto reporter_view = reporters.view();
        auto grad_counts   = reporter_gradient_offsets_counts.counts();
        auto hess_counts   = reporter_hessian_offsets_counts.counts();

        for(auto&& R : reporter_view)
        {
            ReportExtentInfo extent_info;
            extent_info.m_gradient_only = info.gradient_only();
            R->report_extent(extent_info);

            grad_counts[R->m_index] = extent_info.m_gradient_count;
            hess_counts[R->m_index] = extent_info.m_hessian_count;
        }

        reporter_gradient_offsets_counts.scan();
        reporter_hessian_offsets_counts.scan();

        if(!info.gradient_only())
            H12x12_count += reporter_hessian_offsets_counts.total_count();
    }


    if(dytopo_effect_receiver && !info.gradient_only())
    {
        H12x12_count += dytopo_effect_receiver->hessians().triplet_count();
    }


    auto H3x3_count = H12x12_count * (4 * 4);

    // Debug: check whether kinetic+shape (Complement part) is considered.
    // Print only once per process to avoid log spam.
    static bool printed = false;
    if(!printed)
    {
        logger::info("[debug] ABDLinearSubsystem::report_extent "
                     "component_flags={}, gradient_only={}, has_complement={}, "
                     "H12x12_count={}, H3x3_count={}, dof_count={}",
                     enum_flags_name(info.component_flags()),
                     info.gradient_only(),
                     has_complement,
                     H12x12_count,
                     H3x3_count,
                     dof_count);
        printed = true;
    }

    if(info.gradient_only())
    {
        UIPC_ASSERT(H3x3_count == 0,
                    "Hessian block count should be zero (got {}) when gradient_only is true",
                    H3x3_count);
    }

    info.extent(H3x3_count, dof_count);
}

void ABDLinearSubsystem::Impl::assemble(GlobalLinearSystem::DiagInfo& info)
{
    using namespace muda;
    const bool corex_trace = (std::getenv("UIPC_COREX_TRACE_ABD_ASSEMBLE") != nullptr);
    auto trace = [&](const char* msg)
    {
        if(corex_trace)
            logger::info("[corex_trace][abd] {}", msg);
    };

    // 0) Prepare buffers for reporters
    trace("assemble: prepare reporter buffers begin");
    {
        auto N = abd().body_count();

        reporter_gradients.reshape(N);
        reporter_gradients.resize_doublets(reporter_gradient_offsets_counts.total_count());

        reporter_hessians.reshape(N, N);
        reporter_hessians.resize_triplets(reporter_hessian_offsets_counts.total_count());
    }
    trace("assemble: prepare reporter buffers end");

    bool has_complement =
        has_flags(info.component_flags(), GlobalLinearSystem::ComponentFlags::Complement);

    IndexT hess_offset = 0;

    // 1) Static Topo Effect: Kinetic + Shape + Other Reporters
    if(has_complement)
    {
        trace("assemble: kinetic_shape begin");
        _assemble_kinetic_shape(hess_offset, info);
        trace("assemble: kinetic_shape end");
        trace("assemble: reporters begin");
        _assemble_reporters(hess_offset, info);
        trace("assemble: reporters end");
    }
    else  // contact only
    {
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        checkCudaErrors(cudaMemset(info.gradients().buffer_view().data(),
                                   0,
                                   sizeof(Float) * info.gradients().size()));
#else
        info.gradients().buffer_view().fill(0);
#endif
    }

    // 2) Dynamic Topology Effect
    trace("assemble: dytopo begin");
    _assemble_dytopo_effect(hess_offset, info);
    trace("assemble: dytopo end");

    UIPC_ASSERT(hess_offset == info.hessians().triplet_count(),
                "Hessian size mismatch: expected {}, got {}",
                info.hessians().triplet_count(),
                hess_offset);
}

void ABDLinearSubsystem::Impl::_assemble_kinetic_shape(IndexT& hess_offset,
                                                       GlobalLinearSystem::DiagInfo& info)
{
    using namespace muda;
    const bool corex_trace = (std::getenv("UIPC_COREX_TRACE_ABD_ASSEMBLE") != nullptr);
    auto trace = [&](const char* msg)
    {
        if(corex_trace)
            logger::info("[corex_trace][abd] kinetic_shape: {}", msg);
    };
    auto sync_dbg = [&](const char* where)
    {
        if(!corex_trace)
            return;
        auto err = cudaDeviceSynchronize();
        UIPC_ASSERT(err == cudaSuccess,
                    "cudaDeviceSynchronize failed at {}: {}",
                    where,
                    cudaGetErrorString(err));
        logger::info("[corex_trace][abd] kinetic_shape: sync ok at {}", where);
    };

    // Collect Kinetic
    trace("collect kinetic begin");
    ABDLinearSubsystem::ComputeGradientHessianInfo this_info{
        info.gradient_only(), body_id_to_kinetic_gradient, body_id_to_kinetic_hessian, dt};
    abd().kinetic->compute_gradient_hessian(this_info);
    sync_dbg("after kinetic");
    trace("collect kinetic end");

    // Collect Shape
    trace("collect constitutions begin");
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    {
        auto body_count = abd().body_count();
        auto err0 = cudaMemset(body_id_to_shape_gradient.data(),
                               0,
                               sizeof(Vector12) * body_count);
        UIPC_ASSERT(err0 == cudaSuccess,
                    "cudaMemset(shape_gradient) failed: {}",
                    cudaGetErrorString(err0));
        auto err1 = cudaMemset(body_id_to_shape_hessian.data(),
                               0,
                               sizeof(Matrix12x12) * body_count);
        UIPC_ASSERT(err1 == cudaSuccess,
                    "cudaMemset(shape_hessian) failed: {}",
                    cudaGetErrorString(err1));

        for(auto&& [i, cst] : enumerate(abd().constitutions.view()))
        {
            if(corex_trace)
                logger::info("[corex_trace][abd] kinetic_shape: constitution {} uid={} begin",
                             i,
                             cst->uid());

            ABDLinearSubsystem::ComputeGradientHessianInfo this_info{
                info.gradient_only(),
                abd().subview(body_id_to_shape_gradient, cst->m_index),
                abd().subview(body_id_to_shape_hessian, cst->m_index),
                dt};

            cst->compute_gradient_hessian(this_info);

            if(corex_trace)
            {
                auto where = fmt::format("after constitution {} uid={}", i, cst->uid());
                sync_dbg(where.c_str());
                logger::info("[corex_trace][abd] kinetic_shape: constitution {} uid={} end",
                             i,
                             cst->uid());
            }
        }
    }
#else
    for(auto&& [i, cst] : enumerate(abd().constitutions.view()))
    {
        if(corex_trace)
            logger::info("[corex_trace][abd] kinetic_shape: constitution {} begin", i);
        ABDLinearSubsystem::ComputeGradientHessianInfo this_info{
            info.gradient_only(),
            abd().subview(body_id_to_shape_gradient, cst->m_index),
            abd().subview(body_id_to_shape_hessian, cst->m_index),
            dt};

        cst->compute_gradient_hessian(this_info);
        if(corex_trace)
        {
            auto where = fmt::format("after constitution {}", i);
            sync_dbg(where.c_str());
        }
        if(corex_trace)
            logger::info("[corex_trace][abd] kinetic_shape: constitution {} end", i);
    }
#endif
    trace("collect constitutions end");

    trace("assemble gradients kernel begin");
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    {
        int n = static_cast<int>(abd().body_count());
        if(n > 0)
        {
            std::vector<IndexT>  h_is_fixed(n);
            std::vector<IndexT>  h_is_external(n);
            std::vector<Vector12> h_shape(n);
            std::vector<Vector12> h_kinetic(n);
            std::vector<Float>   h_gradients(static_cast<size_t>(n) * 12, 0.0);

            auto copy_or_throw = [](void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, const char* what)
            {
                auto err = cudaMemcpy(dst, src, bytes, kind);
                UIPC_ASSERT(err == cudaSuccess,
                            "cudaMemcpy failed for {}: {}",
                            what,
                            cudaGetErrorString(err));
            };

            copy_or_throw(h_is_fixed.data(),
                          abd().body_id_to_is_fixed.data(),
                          sizeof(IndexT) * static_cast<size_t>(n),
                          cudaMemcpyDeviceToHost,
                          "is_fixed");
            copy_or_throw(h_is_external.data(),
                          abd().body_id_to_external_kinetic.data(),
                          sizeof(IndexT) * static_cast<size_t>(n),
                          cudaMemcpyDeviceToHost,
                          "is_external_kinetic");
            copy_or_throw(h_shape.data(),
                          body_id_to_shape_gradient.data(),
                          sizeof(Vector12) * static_cast<size_t>(n),
                          cudaMemcpyDeviceToHost,
                          "shape_gradient");
            copy_or_throw(h_kinetic.data(),
                          body_id_to_kinetic_gradient.data(),
                          sizeof(Vector12) * static_cast<size_t>(n),
                          cudaMemcpyDeviceToHost,
                          "kinetic_gradient");

            for(int i = 0; i < n; ++i)
            {
                auto base = static_cast<size_t>(i) * 12;
                if(h_is_fixed[i])
                {
                    for(int d = 0; d < 12; ++d)
                        h_gradients[base + d] = 0.0;
                    continue;
                }

                const auto& s = h_shape[i];
                if(!h_is_external[i])
                {
                    const auto& k = h_kinetic[i];
                    for(int d = 0; d < 12; ++d)
                        h_gradients[base + d] = s(d) + k(d);
                }
                else
                {
                    for(int d = 0; d < 12; ++d)
                        h_gradients[base + d] = s(d);
                }

                (void)s;
            }

            copy_or_throw(info.gradients().data(),
                          h_gradients.data(),
                          sizeof(Float) * static_cast<size_t>(n) * 12,
                          cudaMemcpyHostToDevice,
                          "assembled gradients");
        }
    }
#else
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(abd().body_count(),
               [is_fixed = abd().body_id_to_is_fixed.cviewer(),
                is_external_kinetic =
                    abd().body_id_to_external_kinetic.cviewer(),
                shape_gradient = body_id_to_shape_gradient.cviewer(),
                kinetic_gradient = body_id_to_kinetic_gradient.cviewer(),
                gradients = info.gradients().viewer()] __device__(int i) mutable
               {
                   constexpr int N = 12;
                   auto          base = i * N;
                   if(is_fixed(i))
                   {
                       for(int k = 0; k < N; ++k)
                           gradients(base + k) = 0.0;
                       return;
                   }

                   auto shape = shape_gradient(i);
                   if(!is_external_kinetic(i)) [[likely]]
                   {
                       auto kin = kinetic_gradient(i);
                       for(int k = 0; k < N; ++k)
                           gradients(base + k) = shape(k) + kin(k);
                   }
                   else
                   {
                       for(int k = 0; k < N; ++k)
                           gradients(base + k) = shape(k);
                   }
               });
#endif
    trace("assemble gradients kernel end");

    if(info.gradient_only())
        return;

    auto body_count = body_id_to_shape_hessian.size();
    auto H3x3_count = body_count * (4 * 4);
    auto body_H3x3  = info.hessians().subview(hess_offset, H3x3_count);

    trace("assemble hessians kernel begin");
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    {
        int n = static_cast<int>(body_count);
        if(n > 0)
        {
            std::vector<IndexT>     h_is_fixed(n);
            std::vector<IndexT>     h_is_external(n);
            std::vector<Matrix12x12> h_shape(n);
            std::vector<Matrix12x12> h_kinetic(n);
            std::vector<Matrix12x12> h_diag(n);

            checkCudaErrors(cudaMemcpy(h_is_fixed.data(),
                                       abd().body_id_to_is_fixed.data(),
                                       sizeof(IndexT) * n,
                                       cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_is_external.data(),
                                       abd().body_id_to_external_kinetic.data(),
                                       sizeof(IndexT) * n,
                                       cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_shape.data(),
                                       body_id_to_shape_hessian.data(),
                                       sizeof(Matrix12x12) * n,
                                       cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_kinetic.data(),
                                       body_id_to_kinetic_hessian.data(),
                                       sizeof(Matrix12x12) * n,
                                       cudaMemcpyDeviceToHost));

            constexpr int BLK = 3;
            int triplets_per_body = 4 * 4;
            int total_triplets    = n * triplets_per_body;
            std::vector<int>     h_rows(total_triplets);
            std::vector<int>     h_cols(total_triplets);
            std::vector<Matrix3x3> h_vals(total_triplets);

            for(int I = 0; I < n; ++I)
            {
                Matrix12x12 H12x12;
                if(h_is_fixed[I])
                {
                    H12x12.setIdentity();
                }
                else
                {
                    H12x12 = h_shape[I];
                    if(!h_is_external[I])
                        H12x12 += h_kinetic[I];
                }
                h_diag[I] = H12x12;

                zero_out_lower(H12x12);

                int base_triplet = I * triplets_per_body;
                for(int ii = 0; ii < 4; ++ii)
                {
                    for(int jj = 0; jj < 4; ++jj)
                    {
                        int idx       = base_triplet + ii * 4 + jj;
                        h_rows[idx]   = I * 4 + ii;
                        h_cols[idx]   = I * 4 + jj;
                        h_vals[idx]   = H12x12.template block<BLK, BLK>(ii * BLK, jj * BLK);
                    }
                }
            }

            checkCudaErrors(cudaMemcpy(this->diag_hessian.data(),
                                       h_diag.data(),
                                       sizeof(Matrix12x12) * n,
                                       cudaMemcpyHostToDevice));

            auto dst_rows = body_H3x3.row_indices();
            auto dst_cols = body_H3x3.col_indices();
            auto dst_vals = body_H3x3.values();

            if(std::getenv("UIPC_COREX_TRACE_LINEAR_SYSTEM"))
            {
                logger::info("[corex_trace][abd] hess host: n={}, triplets={}, "
                             "dst_rows.data()={}, dst_cols.data()={}, dst_vals.data()={}",
                             n, total_triplets,
                             (void*)dst_rows.data(), (void*)dst_cols.data(), (void*)dst_vals.data());
            }

            checkCudaErrors(cudaMemcpy(dst_rows.data(),
                                       h_rows.data(),
                                       sizeof(int) * total_triplets,
                                       cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(dst_cols.data(),
                                       h_cols.data(),
                                       sizeof(int) * total_triplets,
                                       cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(dst_vals.data(),
                                       h_vals.data(),
                                       sizeof(Matrix3x3) * total_triplets,
                                       cudaMemcpyHostToDevice));
            checkCudaErrors(cudaDeviceSynchronize());
        }
    }
#else
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(body_count,
               [dst      = body_H3x3.viewer().name("dst_hessian"),
                is_fixed = abd().body_id_to_is_fixed.cviewer().name("is_fixed"),
                is_external_kinetic =
                    abd().body_id_to_external_kinetic.cviewer().name("external_kinetic"),
                shape_hessian = body_id_to_shape_hessian.cviewer().name("src_hessian"),
                kinetic_hessian = body_id_to_kinetic_hessian.cviewer().name("kinetic_hessian"),
                diag_hessian = this->diag_hessian.viewer().name("diag_hessian")] __device__(int I) mutable
               {
                   TripletMatrixUnpacker MA{dst};
                   Matrix12x12           H12x12;

                   if(is_fixed(I))
                   {
                       // Fill kinetic hessian to identity to avoid singularity
                       H12x12.setIdentity();
                   }
                   else
                   {
                       // if not fixed, fill shape hessian
                       H12x12 = shape_hessian(I);

                       // if not external kinetic, add kinetic gradient
                       if(!is_external_kinetic(I)) [[likely]]
                       {
                           H12x12 += kinetic_hessian(I);
                       }
                   }

                   // record diagonal hessian for diag-inv preconditioner
                   diag_hessian(I) = H12x12;

                   // set the lower triangle blocks to zero for robustness
                   zero_out_lower(H12x12);

                   MA.block<4, 4>(I * 4 * 4)  // triplet range of [I*4*4, (I+1)*4*4)
                       .write(I * 4,          // begin row
                              I * 4,          // begin col
                              H12x12);
               });
#endif
    trace("assemble hessians kernel end");

    hess_offset += H3x3_count;
    trace("kinetic_shape done");
}

void ABDLinearSubsystem::Impl::_assemble_reporters(IndexT& offset,
                                                   GlobalLinearSystem::DiagInfo& info)
{
    using namespace muda;

    // Fill TripletMatrix and DoubletVector
    for(auto& R : reporters.view())
    {
        AssembleInfo assemble_info{this, R->m_index, info.gradient_only()};
        R->assemble(assemble_info);
    }

    if(reporter_gradients.doublet_count())
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(reporter_gradients.doublet_count(),
                   [dst = info.gradients().viewer().name("dst_gradient"),
                    src = reporter_gradients.cviewer().name("src_gradient"),
                    is_fixed = abd().body_id_to_is_fixed.cviewer().name(
                        "is_fixed")] __device__(int I) mutable
                   {
                       auto&& [body_i, G12] = src(I);

                       if(is_fixed(body_i))
                       {
                           // Do nothing
                       }
                       else
                       {
                           dst.segment<12>(body_i * 12).atomic_add(G12);
                       }
                   });
    }

    if(!info.gradient_only() && reporter_hessians.triplet_count())
    {
        // get rest
        auto H3x3s = info.hessians().subview(offset);

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(reporter_hessians.triplet_count(),
                   [dst = H3x3s.viewer().name("dst_hessian"),
                    src = reporter_hessians.cviewer().name("src_hessian"),
                    diag_hessian = this->diag_hessian.viewer().name("diag_hessian"),
                    is_fixed = abd().body_id_to_is_fixed.cviewer().name(
                        "is_fixed")] __device__(int I) mutable
                   {
                       TripletMatrixUnpacker MU{dst};
                       Matrix12x12           H12x12;
                       auto&& [body_i, body_j, Value] = src(I);
                       H12x12                         = Value;

                       bool has_fixed = (is_fixed(body_i) || is_fixed(body_j));

                       // Fill diagonal hessian for diag-inv preconditioner
                       if(body_i == body_j && !has_fixed)
                       {
                           eigen::atomic_add(diag_hessian(body_i), H12x12);
                       }

                       if(has_fixed)
                       {
                           // Zero out hessian for fixed bodies
                           H12x12.setZero();
                       }
                       else
                       {
                           if(body_i == body_j)
                           {
                               // Since body_i == body_j, we only fill the upper triangle part
                               zero_out_lower(H12x12);
                           }
                           else if(body_i > body_j)
                           {
                               // If all the reporters only report upper triangle part, this branch should not be hit
                               H12x12.setZero();
                           }
                       }

                       MU.block<4, 4>(I * 4 * 4)  // triplet range of [I*4*4, (I+1)*4*4)
                           .write(body_i * 4,  // begin row
                                  body_j * 4,  // begin col
                                  H12x12);
                   });

        offset += reporter_hessians.triplet_count() * (4 * 4);
    }
}

void ABDLinearSubsystem::Impl::_assemble_dytopo_effect(IndexT& offset,
                                                       GlobalLinearSystem::DiagInfo& info)
{
    using namespace muda;

    auto  vertex_offset = affine_body_vertex_reporter->vertex_offset();
    SizeT dytopo_effect_gradient_count = 0;
    if(dytopo_effect_receiver)
    {
        dytopo_effect_gradient_count =
            dytopo_effect_receiver->gradients().doublet_count();
    }

    if(dytopo_effect_gradient_count)
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(dytopo_effect_gradient_count,
                   [dytopo_effect_gradient =
                        dytopo_effect_receiver->gradients().cviewer().name("dytopo_effect_gradient"),
                    gradients = info.gradients().viewer().name("gradients"),
                    v2b = abd().vertex_id_to_body_id.cviewer().name("v2b"),
                    Js  = abd().vertex_id_to_J.cviewer().name("Js"),
                    is_fixed = abd().body_id_to_is_fixed.cviewer().name("is_fixed"),
                    vertex_offset = vertex_offset,
                    cout = KernelCout::viewer()] __device__(int I) mutable
                   {
                       const auto& [g_i, G3] = dytopo_effect_gradient(I);

                       auto  i      = g_i - vertex_offset;
                       auto  body_i = v2b(i);
                       auto& J_i    = Js(i);

                       if(is_fixed(body_i))
                       {
                           // Do nothing
                       }
                       else
                       {
                           Vector12 G12 = J_i.T() * G3;
                           gradients.segment<12>(body_i * 12).atomic_add(G12);

                           // cout << "DG(" << I << "): " << G12.transpose().eval() << "\n";
                       }
                   });
    }

    if(info.gradient_only())
        return;

    SizeT dytopo_effect_hessian_count = 0;
    if(dytopo_effect_receiver)
        dytopo_effect_hessian_count = dytopo_effect_receiver->hessians().triplet_count();

    auto H3x3_count         = dytopo_effect_hessian_count * (4 * 4);
    auto dytopo_effect_H3x3 = info.hessians().subview(offset, H3x3_count);

    if(dytopo_effect_hessian_count)
    {
        // Half Contact Hessian
        // ref: https://github.com/spiriMirror/libuipc/issues/272
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(dytopo_effect_hessian_count,
                   [dytopo_effect_hessian =
                        dytopo_effect_receiver->hessians().cviewer().name("dytopo_effect_hessian"),
                    dst = dytopo_effect_H3x3.viewer().name("dst_hessian"),
                    v2b = abd().vertex_id_to_body_id.cviewer().name("v2b"),
                    Js  = abd().vertex_id_to_J.cviewer().name("Js"),
                    is_fixed = abd().body_id_to_is_fixed.cviewer().name("is_fixed"),
                    diag_hessian = this->diag_hessian.viewer().name("diag_hessian"),
                    vertex_offset = vertex_offset] __device__(int I) mutable
                   {
                       const auto& [g_i, g_j, H3x3] = dytopo_effect_hessian(I);

                       auto i = g_i - vertex_offset;
                       auto j = g_j - vertex_offset;

                       auto body_i = v2b(i);
                       auto body_j = v2b(j);

                       auto& J_i = Js(i);
                       auto& J_j = Js(j);

                       Matrix12x12 H12x12;

                       // We know half contact hessian i <= j
                       // but we don't know body_i and body_j order
                       // so test and swap if necessary
                       IndexT L = body_i;
                       IndexT R = body_j;
                       if(body_i > body_j)
                       {
                           L = body_j;
                           R = body_i;
                       }

                       if(is_fixed(body_i) || is_fixed(body_j))
                       {
                           H12x12.setZero();
                       }
                       else
                       {
                           if(body_i < body_j)
                           {
                               H12x12 = ABDJacobi::JT_H_J(J_i.T(), H3x3, J_j);
                           }
                           else if(body_i > body_j)
                           {
                               H12x12 = ABDJacobi::JT_H_J(J_j.T(), H3x3.transpose(), J_i);
                           }
                           else  // body_i == body_j
                           {
                               // Two vertices from the same body
                               if(i != j)
                               {
                                   H12x12 = ABDJacobi::JT_H_J(J_i.T(), H3x3, J_j)
                                            + ABDJacobi::JT_H_J(J_j.T(), H3x3.transpose(), J_i);
                               }
                               else  // i == j
                               {
                                   H12x12 = ABDJacobi::JT_H_J(J_i.T(), H3x3, J_j);
                               }

                               // Fill diagonal hessian for diag-inv preconditioner
                               eigen::atomic_add(diag_hessian(body_i), H12x12);

                               // Since body_i == body_j, we only fill the upper triangle part
                               zero_out_lower(H12x12);
                           }
                       }

                       TripletMatrixUnpacker MU{dst};
                       MU.block<4, 4>(I * 4 * 4)  // triplet range of [I*16, (I+1)*16)
                           .write(L * 4,          // begin row
                                  R * 4,          // begin col
                                  H12x12);
                   });
    }

    offset += H3x3_count;
}

void ABDLinearSubsystem::Impl::accuracy_check(GlobalLinearSystem::AccuracyInfo& info)
{
    info.satisfied(true);
}

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
__global__ void kernel_retrieve_solution(int n, Vector12* dq, const Float* x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    for(int d = 0; d < 12; ++d)
        dq[i](d) = -x[i * 12 + d];
}
#endif

void ABDLinearSubsystem::Impl::retrieve_solution(GlobalLinearSystem::SolutionInfo& info)
{
    using namespace muda;

    auto dq = abd().body_id_to_dq.view();
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    int n = static_cast<int>(abd().body_count());
    if(n > 0)
    {
        int block = 128;
        int grid  = (n + block - 1) / block;
        kernel_retrieve_solution<<<grid, block>>>(
            n, (Vector12*)dq.data(), (const Float*)info.solution().data());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        {
            std::vector<Vector12> h_dq(n);
            std::vector<Float> h_x(n * 12);
            checkCudaErrors(cudaMemcpy(h_dq.data(), dq.data(), n * sizeof(Vector12), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_x.data(), info.solution().data(), n * 12 * sizeof(Float), cudaMemcpyDeviceToHost));
            for(int i = 0; i < n; ++i)
            {
                fprintf(stderr, "[retrieve] body %d: dq=[%.6f %.6f %.6f | %.6f %.6f %.6f | %.6f %.6f %.6f | %.6f %.6f %.6f]\n",
                    i, h_dq[i](0), h_dq[i](1), h_dq[i](2),
                    h_dq[i](3), h_dq[i](4), h_dq[i](5),
                    h_dq[i](6), h_dq[i](7), h_dq[i](8),
                    h_dq[i](9), h_dq[i](10), h_dq[i](11));
                fprintf(stderr, "           x  =[%.6f %.6f %.6f | %.6f %.6f %.6f | %.6f %.6f %.6f | %.6f %.6f %.6f]\n",
                    h_x[i*12+0], h_x[i*12+1], h_x[i*12+2],
                    h_x[i*12+3], h_x[i*12+4], h_x[i*12+5],
                    h_x[i*12+6], h_x[i*12+7], h_x[i*12+8],
                    h_x[i*12+9], h_x[i*12+10], h_x[i*12+11]);
            }
        }
    }
#else
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(abd().body_count(),
               [dq = dq.viewer().name("dq"),
                x = info.solution().viewer().name("x")] __device__(int i) mutable
               {
                   dq(i) = -x.segment<12>(i * 12).as_eigen();
               });
#endif
}

Float ABDLinearSubsystem::Impl::diag_norm()
{
    auto diag_hess = diag_hessian.view();
    block_norm.resize(diag_hess.size() * 12);
    muda::ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(diag_hess.size(),
               [diag_hess        = diag_hess.cviewer().name("diag_hess"),
                diag_blocks_norm = block_norm.viewer().name("diag_blocks_norm"),
                is_fixed = abd().body_id_to_is_fixed.cviewer().name("is_fixed")] __device__(int idx) mutable
               {
                   for(int i = 0; i < 12; i++)
                       diag_blocks_norm(idx * 12 + i) =
                           is_fixed(idx) ? 0 : abs(diag_hess(idx)(i, i));
               });

    muda::DeviceReduce().Max(block_norm.data(), reduced_norm.data(), block_norm.size());

    return reduced_norm;
}

Float ABDLinearSubsystem::Impl::mass_norm()
{
    auto mass = abd().body_id_to_abd_mass.view();
    block_norm.resize(mass.size());
    muda::ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(mass.size(),
               [mass       = mass.cviewer().name("diag_hess"),
                block_norm = block_norm.viewer().name("diag_blocks_norm"),
                is_fixed = abd().body_id_to_is_fixed.cviewer().name("is_fixed")] __device__(int idx) mutable
               { block_norm(idx) = is_fixed(idx) ? 0 : mass(idx).mass(); });

    muda::DeviceReduce().Max(block_norm.data(), reduced_norm.data(), block_norm.size());

    return reduced_norm;
}
}  // namespace uipc::backend::cuda

namespace uipc::backend::cuda
{
void ABDLinearSubsystem::do_init(InitInfo& info)
{
    m_impl.init();
}

void ABDLinearSubsystem::do_report_extent(GlobalLinearSystem::DiagExtentInfo& info)
{
    m_impl.report_extent(info);
}

void ABDLinearSubsystem::do_assemble(GlobalLinearSystem::DiagInfo& info)
{
    m_impl.assemble(info);
}

void ABDLinearSubsystem::do_accuracy_check(GlobalLinearSystem::AccuracyInfo& info)
{
    m_impl.accuracy_check(info);
}

void ABDLinearSubsystem::do_retrieve_solution(GlobalLinearSystem::SolutionInfo& info)
{
    m_impl.retrieve_solution(info);
}

Float ABDLinearSubsystem::do_diag_norm(GlobalLinearSystem::DiagNormInfo& info)
{
    return m_impl.diag_norm();
}

Float ABDLinearSubsystem::do_mass_norm(GlobalLinearSystem::DiagNormInfo& info)
{
    return m_impl.mass_norm();
}

U64 ABDLinearSubsystem::get_uid() const noexcept
{
    return ABDLinearSubsystemUID;
}

void ABDLinearSubsystem::add_reporter(ABDLinearSubsystemReporter* reporter)
{
    UIPC_ASSERT(reporter, "reporter cannot be null");
    check_state(SimEngineState::BuildSystems, "add_reporter");
    m_impl.reporters.register_sim_system(*reporter);
}

void ABDLinearSubsystem::do_report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info)
{
    m_impl.report_init_extent(info);
}

void ABDLinearSubsystem::do_receive_init_dof_info(GlobalLinearSystem::InitDofInfo& info)
{
    m_impl.receive_init_dof_info(world(), info);
}

ABDLinearSubsystem::AssembleInfo::AssembleInfo(Impl* impl, IndexT index, bool gradient_only) noexcept
    : m_impl(impl)
    , m_index(index)
    , m_gradient_only(gradient_only)
{
}

muda::DoubletVectorView<Float, 12> ABDLinearSubsystem::AssembleInfo::gradients() const
{
    auto [offset, count] = m_impl->reporter_gradient_offsets_counts[m_index];
    return m_impl->reporter_gradients.view().subview(offset, count);
}

muda::TripletMatrixView<Float, 12, 12> ABDLinearSubsystem::AssembleInfo::hessians() const
{
    auto [offset, count] = m_impl->reporter_hessian_offsets_counts[m_index];
    return m_impl->reporter_hessians.view().subview(offset, count);
}

bool ABDLinearSubsystem::AssembleInfo::gradient_only() const noexcept
{
    return m_gradient_only;
}

void ABDLinearSubsystem::ReportExtentInfo::gradient_count(SizeT size)
{
    m_gradient_count = size;
}

void ABDLinearSubsystem::ReportExtentInfo::hessian_count(SizeT size)
{
    m_hessian_count = size;
}

void ABDLinearSubsystem::ReportExtentInfo::check(std::string_view name) const
{
    check_report_extent(m_gradient_only_checked, m_gradient_only, m_hessian_count, name);
}

AffineBodyDynamics::Impl& ABDLinearSubsystem::Impl::abd() const noexcept
{
    return affine_body_dynamics->m_impl;
}
}  // namespace uipc::backend::cuda
