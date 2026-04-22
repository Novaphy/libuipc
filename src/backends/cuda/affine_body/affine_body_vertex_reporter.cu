#include <affine_body/affine_body_vertex_reporter.h>
#include <global_geometry/global_vertex_manager.h>
#include <affine_body/affine_body_body_reporter.h>
#include <uipc/builtin/attribute_name.h>
#include <muda/check/check_cuda_errors.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(AffineBodyVertexReporter);

constexpr static U64 AffineBodyVertexReporterUID = 0;

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
__global__ void kernel_abd_report_displacements(int N,
                                                const IndexT* v2b,
                                                const ABDJacobi* Js,
                                                const Vector12* dqs,
                                                Vector3* displacements)
{
    int vI = blockIdx.x * blockDim.x + threadIdx.x;
    if(vI >= N) return;
    auto body_id         = v2b[vI];
    const Vector12& dq   = dqs[body_id];
    const ABDJacobi& J   = Js[vI];
    displacements[vI]    = J * dq;
}
#endif

void AffineBodyVertexReporter::do_build(BuildInfo& info)
{
    m_impl.affine_body_dynamics = &require<AffineBodyDynamics>();
    m_impl.body_reporter        = &require<AffineBodyBodyReporter>();
}

void AffineBodyVertexReporter::request_attribute_update() noexcept
{
    m_impl.require_update_attributes = true;
}

void AffineBodyVertexReporter::Impl::report_count(VertexCountInfo& info)
{
    info.count(abd().h_vertex_id_to_J.size());
}

void AffineBodyVertexReporter::Impl::init_attributes(VertexAttributeInfo& info)
{
    using namespace muda;

    auto N = info.positions().size();

    UIPC_ASSERT(body_reporter->body_offset() >= 0,
                "AffineBodyBodyReporter is not ready, body_offset={}, lifecycle issue?",
                body_reporter->body_offset());

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    {
        int n = static_cast<int>(N);
        int body_offset = body_reporter->body_offset();

        std::vector<ABDJacobi> h_J(n);
        std::vector<IndexT>    h_v2b(n);
        std::vector<Vector12>  h_q(abd().body_count());

        cudaMemcpy(h_J.data(), abd().vertex_id_to_J.data(), n*sizeof(ABDJacobi), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_v2b.data(), abd().vertex_id_to_body_id.data(), n*sizeof(IndexT), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_q.data(), abd().body_id_to_q.data(), abd().body_count()*sizeof(Vector12), cudaMemcpyDeviceToHost);

        std::vector<IndexT>  h_coindices(n);
        std::vector<Vector3> h_pos(n), h_rest(n);
        std::vector<IndexT>  h_dst_v2b(n);

        for(int i = 0; i < n; ++i)
        {
            h_coindices[i] = i;
            auto body_id = h_v2b[i];
            h_pos[i] = h_J[i].point_x(h_q[body_id]);
            h_rest[i] = h_J[i].x_bar();
            h_dst_v2b[i] = body_id + body_offset;
            spdlog::info("[INIT_ATTR] v{} body={} pos=({},{},{}) rest=({},{},{})",
                i, body_id, h_pos[i][0], h_pos[i][1], h_pos[i][2],
                h_rest[i][0], h_rest[i][1], h_rest[i][2]);
        }

        cudaMemcpy((void*)info.coindices().data(), h_coindices.data(), n*sizeof(IndexT), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)info.positions().data(), h_pos.data(), n*sizeof(Vector3), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)info.rest_positions().data(), h_rest.data(), n*sizeof(Vector3), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)info.body_ids().data(), h_dst_v2b.data(), n*sizeof(IndexT), cudaMemcpyHostToDevice);
    }
#else
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(N,
               [coindices   = info.coindices().viewer().name("coindices"),
                src_pos     = abd().vertex_id_to_J.cviewer().name("src_pos"),
                dst_pos     = info.positions().viewer().name("dst_pos"),
                v2b         = abd().vertex_id_to_body_id.cviewer().name("v2b"),
                body_offset = body_reporter->body_offset(),
                dst_v2b     = info.body_ids().viewer().name("dst_v2b"),
                qs          = abd().body_id_to_q.cviewer().name("qs"),
                dst_rest_pos = info.rest_positions().viewer().name("rest_pos")] __device__(int i) mutable
               {
                   coindices(i) = i;

                   auto        body_id = v2b(i);
                   const auto& q       = qs(body_id);
                   dst_pos(i)          = src_pos(i).point_x(q);
                   dst_rest_pos(i)     = src_pos(i).x_bar();
                   dst_v2b(i) = body_id + body_offset;
               });
#endif

    auto async_copy = []<typename T>(span<T> src, muda::BufferView<T> dst)
    { muda::BufferLaunch().copy<T>(dst, src.data()); };

    async_copy(span{abd().h_vertex_id_to_contact_element_id}, info.contact_element_ids());
    async_copy(span{abd().h_vertex_id_to_subscene_contact_element_id},
               info.subscene_element_ids());
    async_copy(span{abd().h_vertex_id_to_d_hat}, info.d_hats());
}

void AffineBodyVertexReporter::Impl::update_attributes(VertexAttributeInfo& info)
{
    using namespace muda;

    auto N = info.positions().size();

#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    {
        int n = static_cast<int>(N);
        std::vector<ABDJacobi> h_J(n);
        std::vector<IndexT>    h_v2b(n);
        std::vector<Vector12>  h_q(abd().body_count());
        std::vector<IndexT>    h_coindices(n);
        std::vector<Vector3>   h_pos(n);

        cudaMemcpy(h_J.data(), abd().vertex_id_to_J.data(), n*sizeof(ABDJacobi), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_v2b.data(), abd().vertex_id_to_body_id.data(), n*sizeof(IndexT), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_q.data(), abd().body_id_to_q.data(), abd().body_count()*sizeof(Vector12), cudaMemcpyDeviceToHost);

        for(int i = 0; i < n; ++i)
        {
            h_coindices[i] = i;
            auto body_id = h_v2b[i];
            h_pos[i] = h_J[i].point_x(h_q[body_id]);
        }

        cudaMemcpy((void*)info.coindices().data(), h_coindices.data(), n*sizeof(IndexT), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)info.positions().data(), h_pos.data(), n*sizeof(Vector3), cudaMemcpyHostToDevice);
    }
#else
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(N,
               [coindices = info.coindices().viewer().name("coindices"),
                src_pos   = abd().vertex_id_to_J.cviewer().name("src_pos"),
                dst_pos   = info.positions().viewer().name("dst_pos"),
                v2b       = abd().vertex_id_to_body_id.cviewer().name("v2b"),
                qs = abd().body_id_to_q.cviewer().name("qs")] __device__(int i) mutable
               {
                   coindices(i)        = i;
                   auto        body_id = v2b(i);
                   const auto& q       = qs(body_id);
                   dst_pos(i)          = src_pos(i).point_x(q);
               });
#endif

    info.require_discard_friction();
}

void AffineBodyVertexReporter::Impl::report_displacements(VertexDisplacementInfo& info)
{
    using namespace muda;
    auto N = info.coindices().size();
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    if(N > 0)
    {
        int n              = static_cast<int>(N);
        constexpr int block = 256;
        int grid           = (n + block - 1) / block;
        kernel_abd_report_displacements<<<grid, block>>>(n,
                                                          abd().vertex_id_to_body_id.data(),
                                                          abd().vertex_id_to_J.data(),
                                                          abd().body_id_to_dq.data(),
                                                          info.displacements().data());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }
#else
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(N,
               [coindices = info.coindices().viewer().name("coindices"),
                displacements = info.displacements().viewer().name("displacements"),
                v2b = abd().vertex_id_to_body_id.cviewer().name("v2b"),
                dqs = abd().body_id_to_dq.cviewer().name("dqs"),
                Js = abd().vertex_id_to_J.cviewer().name("Js")] __device__(int vI) mutable
               {
                   auto             body_id = v2b(vI);
                   const Vector12&  dq      = dqs(body_id);
                   const ABDJacobi& J       = Js(vI);
                   auto&            dx      = displacements(vI);
                   dx                       = J * dq;
               });
#endif
}

void AffineBodyVertexReporter::do_report_count(VertexCountInfo& info)
{
    m_impl.report_count(info);
}

void AffineBodyVertexReporter::do_report_attributes(VertexAttributeInfo& info)
{
    if(info.frame() == 0)
    {
        auto global_offset = info.coindices().offset();

        auto geo_slots = world().scene().geometries();

        // add global vertex offset attribute
        m_impl.affine_body_dynamics->for_each(  //
            geo_slots,
            [&](const AffineBodyDynamics::ForEachInfo& I, geometry::SimplicialComplex& sc)
            {
                auto gvo = sc.meta().find<IndexT>(builtin::global_vertex_offset);
                if(!gvo)
                {
                    gvo = sc.meta().create<IndexT>(builtin::global_vertex_offset);
                }

                // [global-vertex-offset] = [vertex-offset-in-abd-system] + [abd-system-vertex-offset]
                view(*gvo)[0] = I.geo_info().vertex_offset + global_offset;
            });

        m_impl.init_attributes(info);
    }
    else
    {
        if(m_impl.require_update_attributes)
        {
            m_impl.update_attributes(info);
            m_impl.require_update_attributes = false;
        }
    }
}

void AffineBodyVertexReporter::do_report_displacements(VertexDisplacementInfo& info)
{
    m_impl.report_displacements(info);
}

U64 AffineBodyVertexReporter::get_uid() const noexcept
{
    return AffineBodyVertexReporterUID;
}
}  // namespace uipc::backend::cuda
