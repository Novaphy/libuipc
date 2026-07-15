#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#include <affine_body/affine_body_vertex_reporter.h>
#include <global_geometry/global_vertex_manager.h>
#include <affine_body/affine_body_body_reporter.h>
#include <uipc/builtin/attribute_name.h>
#include <muda/check/check_cuda_errors.h>
#include <cstdlib>
#include <spdlog/spdlog.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(AffineBodyVertexReporter);

constexpr static U64 AffineBodyVertexReporterUID = 0;

MUDA_DEVICE Vector3 corex_abd_point_x(const ABDJacobi& J, const Vector12& q)
{
    const Vector3& x = J.x_bar();
    Vector3        ret;
    ret[0] = x[0] * q[3] + x[1] * q[4] + x[2] * q[5] + q[0];
    ret[1] = x[0] * q[6] + x[1] * q[7] + x[2] * q[8] + q[1];
    ret[2] = x[0] * q[9] + x[1] * q[10] + x[2] * q[11] + q[2];
    return ret;
}

__global__ void kernel_abd_init_vertex_attributes(int            n,
                                                  IndexT         body_offset,
                                                  const ABDJacobi* __restrict__ src_J,
                                                  const IndexT* __restrict__ v2b,
                                                  const Vector12* __restrict__ qs,
                                                  IndexT* __restrict__     coindices,
                                                  Vector3* __restrict__    dst_pos,
                                                  Vector3* __restrict__    dst_rest,
                                                  IndexT* __restrict__     dst_body_ids)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    coindices[i]   = i;
    IndexT body_id = v2b[i];
    dst_pos[i]     = corex_abd_point_x(src_J[i], qs[body_id]);
    dst_rest[i]    = src_J[i].x_bar();
    dst_body_ids[i] = body_id + body_offset;
}

__global__ void kernel_abd_update_vertex_positions(int            n,
                                                   const ABDJacobi* __restrict__ src_J,
                                                   const IndexT* __restrict__    v2b,
                                                   const Vector12* __restrict__  qs,
                                                   IndexT* __restrict__         coindices,
                                                   Vector3* __restrict__        dst_pos)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    coindices[i]   = i;
    IndexT body_id = v2b[i];
    dst_pos[i]     = corex_abd_point_x(src_J[i], qs[body_id]);
}

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

    {
        int n           = static_cast<int>(N);
        int body_offset = body_reporter->body_offset();

        if(n > 0)
        {
            constexpr int block = 256;
            int           grid  = (n + block - 1) / block;
            kernel_abd_init_vertex_attributes<<<grid, block>>>(n,
                                                               body_offset,
                                                               abd().vertex_id_to_J.data(),
                                                               abd().vertex_id_to_body_id.data(),
                                                               abd().body_id_to_q.data(),
                                                               info.coindices().data(),
                                                               info.positions().data(),
                                                               info.rest_positions().data(),
                                                               info.body_ids().data());
            checkCudaErrors(cudaGetLastError());
        }
    }

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

    {
        int n = static_cast<int>(N);
        if(n > 0)
        {
            constexpr int block = 256;
            int           grid  = (n + block - 1) / block;
            kernel_abd_update_vertex_positions<<<grid, block>>>(n,
                                                                  abd().vertex_id_to_J.data(),
                                                                  abd().vertex_id_to_body_id.data(),
                                                                  abd().body_id_to_q.data(),
                                                                  info.coindices().data(),
                                                                  info.positions().data());
            checkCudaErrors(cudaGetLastError());
        }
    }

    info.require_discard_friction();
}

void AffineBodyVertexReporter::Impl::report_displacements(VertexDisplacementInfo& info)
{
    using namespace muda;
    auto N = info.coindices().size();
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
        static const bool sync_report =
            std::getenv("UIPC_COREX_ABD_VERTEX_REPORT_SYNC") != nullptr;
        if(sync_report)
            checkCudaErrors(cudaDeviceSynchronize());
    }
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
#else
#include <affine_body/affine_body_vertex_reporter.h>
#include <global_geometry/global_vertex_manager.h>
#include <affine_body/affine_body_body_reporter.h>
#include <uipc/builtin/attribute_name.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(AffineBodyVertexReporter);

constexpr static U64 AffineBodyVertexReporterUID = 0;

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
                   dst_v2b(i) = body_id + body_offset;  // offset by the global body offset
               });

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

    // only update positions
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

    // This update will ruin the friction force computed in previous step, so we need to discard it.
    // ref: https://github.com/spiriMirror/libuipc/issues/303
    info.require_discard_friction();
}

void AffineBodyVertexReporter::Impl::report_displacements(VertexDisplacementInfo& info)
{
    using namespace muda;
    auto N = info.coindices().size();
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
#endif
