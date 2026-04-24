#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#pragma once
// ==============================================================================
// Dual-source whole-file switch (#if Corex / #else NVIDIA upstream).
// Reason: cudafit removed FEM external-force feature (NVIDIA-only); NVIDIA path needs the original declarations/code
// ==============================================================================
#include <type_define.h>
#include <uipc/core/finite_element_state_accessor_feature.h>

namespace uipc::backend::cuda
{
class FiniteElementMethod;
class FiniteElementVertexReporter;

class FiniteElementStateAccessorFeatureOverrider final : public core::FiniteElementStateAccessorFeatureOverrider
{
  public:
    FiniteElementStateAccessorFeatureOverrider(FiniteElementMethod& fem,
                                               FiniteElementVertexReporter& vertex_reporter);

    SizeT get_vertex_count() override;
    void  do_copy_from(const geometry::SimplicialComplex& state_geo) override;
    void  do_copy_to(geometry::SimplicialComplex& state_geo) override;

  private:
    FiniteElementMethod&         m_fem;
    FiniteElementVertexReporter& m_vertex_reporter;
};
}  // namespace uipc::backend::cuda

#else
#pragma once
#include <type_define.h>
#include <uipc/core/finite_element_state_accessor_feature.h>
#include <muda/buffer/device_buffer.h>

namespace uipc::backend::cuda
{
class FiniteElementMethod;
class FiniteElementVertexReporter;

class FiniteElementStateAccessorFeatureOverrider final : public core::FiniteElementStateAccessorFeatureOverrider
{
  public:
    FiniteElementStateAccessorFeatureOverrider(FiniteElementMethod& fem,
                                               FiniteElementVertexReporter& vertex_reporter);

    SizeT get_vertex_count() override;
    void  do_copy_from(const geometry::SimplicialComplex& state_geo) override;
    void  do_copy_to(geometry::SimplicialComplex& state_geo) override;

    void do_copy_position_to(backend::BufferView buffer_view, IndexT vertex_offset, SizeT vertex_count) override;
    void do_copy_velocity_to(backend::BufferView buffer_view, IndexT vertex_offset, SizeT vertex_count) override;

  private:
    FiniteElementMethod&         m_fem;
    FiniteElementVertexReporter& m_vertex_reporter;
};
}  // namespace uipc::backend::cuda
#endif
