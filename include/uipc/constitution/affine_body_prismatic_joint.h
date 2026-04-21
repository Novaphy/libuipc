#pragma once
#include <uipc/constitution/inter_affine_body_constitution.h>
#include <uipc/geometry/simplicial_complex_slot.h>

namespace uipc::constitution
{
class UIPC_CONSTITUTION_API AffineBodyPrismaticJoint final : public InterAffineBodyConstitution
{
  public:
    using SlotTuple =
        std::tuple<S<geometry::SimplicialComplexSlot>, S<geometry::SimplicialComplexSlot>>;

    static Json default_config();

    AffineBodyPrismaticJoint(const Json& config = default_config());

    virtual ~AffineBodyPrismaticJoint();

    /**
     * @brief Apply prismatic joint to edges connecting affine bodies (single-instance mode).
     * 
     * This method assumes each geometry has exactly one instance (instance 0).
     * All joints use the same strength ratio.
     * 
     * @param edges The simplicial complex containing the edges representing the joints.
     * @param l_geo_slots Left geometry slots for each joint.
     * @param r_geo_slots Right geometry slots for each joint.
     * @param strength_ratio The strength ratio of the joint constraint applied to all joints (default: 100).
     */
    void apply_to(geometry::SimplicialComplex&             edges,
                  span<S<geometry::SimplicialComplexSlot>> l_geo_slots,
                  span<S<geometry::SimplicialComplexSlot>> r_geo_slots,
                  Float strength_ratio = Float{100});

    /**
     * @brief Apply prismatic joint to edges connecting affine bodies (multi-instance mode).
     * 
     * This method supports geometries with multiple instances. Each joint can specify
     * which instance of each geometry to connect, and can have its own strength ratio.
     * 
     * @param edges The simplicial complex containing the edges representing the joints.
     * @param l_geo_slots Left geometry slots for each joint.
     * @param l_instance_id Instance IDs for the left geometries (must be in range [0, instances().size())).
     * @param r_geo_slots Right geometry slots for each joint.
     * @param r_instance_id Instance IDs for the right geometries (must be in range [0, instances().size())).
     * @param strength_ratio The strength ratio for each joint (one per edge).
     */
    void apply_to(geometry::SimplicialComplex&             edges,
                  span<S<geometry::SimplicialComplexSlot>> l_geo_slots,
                  span<IndexT>                             l_instance_id,
                  span<S<geometry::SimplicialComplexSlot>> r_geo_slots,
                  span<IndexT>                             r_instance_id,
                  span<Float>                              strength_ratio);

    /**
     * @brief Build a 1-D SimplicialComplex of N edges (2*N vertices) whose
     * endpoints are given by parallel position arrays, and apply the prismatic
     * joint attributes via apply_to().
     */
    geometry::SimplicialComplex create_geometry(
        span<const Vector3>                      position0s,
        span<const Vector3>                      position1s,
        span<S<geometry::SimplicialComplexSlot>> l_geo_slots,
        span<IndexT>                             l_instance_ids,
        span<S<geometry::SimplicialComplexSlot>> r_geo_slots,
        span<IndexT>                             r_instance_ids,
        span<Float>                              strength_ratios);

    /**
     * @brief Build a SimplicialComplex from explicit per-side endpoint
     * positions and apply the prismatic joint attributes. Both sides may
     * supply two endpoints each, allowing distinct anchors per body.
     */
    geometry::SimplicialComplex create_geometry(
        span<const Vector3>                      l_position0,
        span<const Vector3>                      l_position1,
        span<const Vector3>                      r_position0,
        span<const Vector3>                      r_position1,
        span<S<geometry::SimplicialComplexSlot>> l_geo_slots,
        span<IndexT>                             l_instance_ids,
        span<S<geometry::SimplicialComplexSlot>> r_geo_slots,
        span<IndexT>                             r_instance_ids,
        span<Float>                              strength_ratios);

  private:
    virtual U64 get_uid() const noexcept override;
    Json        m_config;
};
}  // namespace uipc::constitution
