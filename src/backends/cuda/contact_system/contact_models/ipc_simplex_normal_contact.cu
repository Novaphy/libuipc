#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#include <contact_system/simplex_normal_contact.h>
#include <contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h>
#include <utils/distance/distance_flagged.h>
#include <utils/codim_thickness.h>
#include <kernel_cout.h>
#include <utils/matrix_assembler.h>
#include <utils/make_spd.h>
#include <utils/primitive_d_hat.h>
#include <pipeline/ipc_pipeline_flag.h>
#include <cstdlib>
#include <vector>
namespace uipc::backend::cuda
{

namespace
{
bool trace_contact_type_energy_enabled()
{
    const char* env = std::getenv("UIPC_COREX_TRACE_CONTACT_TYPE_ENERGY");
    if(!env) return false;
    return env[0] != '\0' && env[0] != '0';
}

bool trace_contact_type_gradient_enabled()
{
    const char* env = std::getenv("UIPC_COREX_TRACE_CONTACT_TYPE_GRAD");
    if(!env) return false;
    return env[0] != '\0' && env[0] != '0';
}

bool trace_barrier_dbdd_enabled()
{
    const char* env = std::getenv("UIPC_COREX_TRACE_BARRIER_DBDD");
    if(!env) return false;
    return env[0] != '\0' && env[0] != '0';
}

bool corex_contact_spd_diag_enabled()
{
    const char* env = std::getenv("UIPC_COREX_CONTACT_SPD_DIAG");
    if(!env) return false;
    return env[0] != '\0' && env[0] != '0';
}

bool corex_contact_input_diag_enabled()
{
    const char* env = std::getenv("UIPC_COREX_CONTACT_INPUT_DIAG");
    return env && env[0] != '\0' && env[0] != '0';
}

bool corex_pe_outlier_diag_enabled()
{
    const char* env = std::getenv("UIPC_COREX_PE_OUTLIER_DIAG");
    return env && env[0] != '\0' && env[0] != '0';
}

Float corex_env_float(const char* name, Float fallback)
{
    const char* env = std::getenv(name);
    if(!env || env[0] == '\0')
        return fallback;
    char*  end = nullptr;
    double v   = std::strtod(env, &end);
    if(end == env)
        return fallback;
    return static_cast<Float>(v);
}

struct CorexContactSpdStats
{
    unsigned int input_diag_enabled;
    unsigned int pe_outlier_diag_enabled;
    Float pe_outlier_ratio_trigger;
    unsigned int count[4];
    unsigned int projected[4];
    Float lower_sum[4];
    Float diag_abs_sum[4];
    Float correction_sum[4];
    Float correction_diag_ratio_sum[4];
    Float max_correction[4];
    Float max_correction_diag_ratio[4];
    unsigned int input_count[4];
    Float kt2_sum[4];
    Float d_hat_sum[4];
    Float thickness_sum[4];
    Float kt2_max[4];
    Float d_hat_max[4];
    Float thickness_max[4];
    unsigned int pe_outlier_count;
    Float pe_outlier_ratio_max;
    Float pe_outlier_corr_max;
    Float pe_outlier_diag_abs;
    Float pe_outlier_lower;
    int pe_outlier_i;
    int pe_outlier_v0;
    int pe_outlier_v1;
    int pe_outlier_v2;
    int pe_outlier_b0;
    int pe_outlier_b1;
    int pe_outlier_b2;
};

enum CorexContactSpdType : int
{
    CorexSpdPT = 0,
    CorexSpdEE = 1,
    CorexSpdPE = 2,
    CorexSpdPP = 3,
};

__device__ inline Float corex_abs_float(Float v)
{
    return v < static_cast<Float>(0) ? -v : v;
}

__device__ inline void corex_atomic_max_float(Float* address, Float value)
{
#if defined(UIPC_FLOAT_SCALAR) && UIPC_FLOAT_SCALAR
    int* address_as_i = reinterpret_cast<int*>(address);
    int  old          = *address_as_i;
    int  assumed;
    do
    {
        assumed = old;
        Float old_v = __int_as_float(assumed);
        if(old_v >= value)
            break;
        old = atomicCAS(address_as_i, assumed, __float_as_int(value));
    } while(assumed != old);
#else
    unsigned long long int* address_as_i =
        reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_i;
    unsigned long long int assumed;
    do
    {
        assumed = old;
        Float old_v = __longlong_as_double(assumed);
        if(old_v >= value)
            break;
        old = atomicCAS(address_as_i, assumed, __double_as_longlong(value));
    } while(assumed != old);
#endif
}

template <int N>
__device__ Float corex_gershgorin_lower(const Matrix<Float, N, N>& H)
{
    Float lower_min = H(0, 0);
    for(int i = 0; i < N; ++i)
    {
        Float off_sum = static_cast<Float>(0);
        for(int j = 0; j < N; ++j)
            if(i != j)
                off_sum += corex_abs_float(H(i, j));
        Float lower = H(i, i) - off_sum;
        if(lower < lower_min)
            lower_min = lower;
    }
    return lower_min;
}

template <int N>
__device__ void corex_record_spd_projection(CorexContactSpdStats* stats,
                                            CorexContactSpdType   type,
                                            const Matrix<Float, N, N>& before,
                                            const Matrix<Float, N, N>& after)
{
    if(!stats)
        return;

    Float lower = corex_gershgorin_lower(before);
    Float diag_abs = static_cast<Float>(0);
    for(int i = 0; i < N; ++i)
        diag_abs += corex_abs_float(before(i, i));
    Float correction = static_cast<Float>(0);
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            correction += corex_abs_float(after(i, j) - before(i, j));
    Float correction_diag_ratio =
        diag_abs > static_cast<Float>(0) ? correction / diag_abs : static_cast<Float>(0);

    int t = static_cast<int>(type);
    atomicAdd(&stats->count[t], 1u);
    atomicAdd(&stats->lower_sum[t], lower);
    atomicAdd(&stats->diag_abs_sum[t], diag_abs);
    atomicAdd(&stats->correction_sum[t], correction);
    atomicAdd(&stats->correction_diag_ratio_sum[t], correction_diag_ratio);
    if(correction > static_cast<Float>(0))
        atomicAdd(&stats->projected[t], 1u);
    corex_atomic_max_float(&stats->max_correction[t], correction);
    corex_atomic_max_float(&stats->max_correction_diag_ratio[t], correction_diag_ratio);
}

template <int N>
__device__ void corex_record_pe_outlier(CorexContactSpdStats* stats,
                                        int                    contact_i,
                                        const Vector3i&        PE,
                                        const Vector3i&        body_ids,
                                        const Matrix<Float, N, N>& before,
                                        const Matrix<Float, N, N>& after)
{
    if(!stats || !stats->pe_outlier_diag_enabled)
        return;

    Float lower = corex_gershgorin_lower(before);
    Float diag_abs = static_cast<Float>(0);
    Float correction = static_cast<Float>(0);
    for(int i = 0; i < N; ++i)
    {
        diag_abs += corex_abs_float(before(i, i));
        for(int j = 0; j < N; ++j)
            correction += corex_abs_float(after(i, j) - before(i, j));
    }
    Float ratio = diag_abs > static_cast<Float>(0)
                    ? correction / diag_abs
                    : static_cast<Float>(0);
    if(ratio >= stats->pe_outlier_ratio_trigger)
        atomicAdd(&stats->pe_outlier_count, 1u);

#if defined(UIPC_FLOAT_SCALAR) && UIPC_FLOAT_SCALAR
    int* address_as_i = reinterpret_cast<int*>(&stats->pe_outlier_ratio_max);
    int  old          = *address_as_i;
    int  assumed;
    do
    {
        assumed = old;
        Float old_v = __int_as_float(assumed);
        if(old_v >= ratio)
            return;
        old = atomicCAS(address_as_i, assumed, __float_as_int(ratio));
    } while(assumed != old);
#else
    unsigned long long int* address_as_i =
        reinterpret_cast<unsigned long long int*>(&stats->pe_outlier_ratio_max);
    unsigned long long int old = *address_as_i;
    unsigned long long int assumed;
    do
    {
        assumed = old;
        Float old_v = __longlong_as_double(assumed);
        if(old_v >= ratio)
            return;
        old = atomicCAS(address_as_i, assumed, __double_as_longlong(ratio));
    } while(assumed != old);
#endif
    stats->pe_outlier_corr_max = correction;
    stats->pe_outlier_diag_abs = diag_abs;
    stats->pe_outlier_lower = lower;
    stats->pe_outlier_i = contact_i;
    stats->pe_outlier_v0 = PE(0);
    stats->pe_outlier_v1 = PE(1);
    stats->pe_outlier_v2 = PE(2);
    stats->pe_outlier_b0 = body_ids(0);
    stats->pe_outlier_b1 = body_ids(1);
    stats->pe_outlier_b2 = body_ids(2);
}

__device__ inline void corex_record_contact_inputs(CorexContactSpdStats* stats,
                                                   CorexContactSpdType   type,
                                                   Float                 kt2,
                                                   Float                 d_hat,
                                                   Float                 thickness)
{
    if(!stats || !stats->input_diag_enabled)
        return;

    int t = static_cast<int>(type);
    atomicAdd(&stats->input_count[t], 1u);
    atomicAdd(&stats->kt2_sum[t], kt2);
    atomicAdd(&stats->d_hat_sum[t], d_hat);
    atomicAdd(&stats->thickness_sum[t], thickness);
    corex_atomic_max_float(&stats->kt2_max[t], corex_abs_float(kt2));
    corex_atomic_max_float(&stats->d_hat_max[t], corex_abs_float(d_hat));
    corex_atomic_max_float(&stats->thickness_max[t], corex_abs_float(thickness));
}

template <int N>
__device__ void corex_apply_spd_conditioning(const Matrix<Float, N, N>& before,
                                             Matrix<Float, N, N>&       after,
                                             Float                      ratio_trigger,
                                             Float                      diag_scale)
{
    if(diag_scale <= static_cast<Float>(0))
        return;

    Float diag_abs = static_cast<Float>(0);
    Float correction = static_cast<Float>(0);
    for(int i = 0; i < N; ++i)
    {
        diag_abs += corex_abs_float(before(i, i));
        for(int j = 0; j < N; ++j)
            correction += corex_abs_float(after(i, j) - before(i, j));
    }
    Float ratio = diag_abs > static_cast<Float>(0)
                    ? correction / diag_abs
                    : static_cast<Float>(0);
    if(ratio < ratio_trigger)
        return;

    Float base = diag_abs / static_cast<Float>(N);
    if(base < static_cast<Float>(1))
        base = static_cast<Float>(1);
    Float shift = base * diag_scale;
    for(int i = 0; i < N; ++i)
        after(i, i) += shift;
}

template <typename EnergyBuffer>
Float host_sum_energy(const EnergyBuffer& energies)
{
    if(energies.size() == 0)
        return static_cast<Float>(0.0);
    std::vector<Float> host(energies.size());
    energies.copy_to(host.data());
    Float sum = static_cast<Float>(0.0);
    for(Float e : host)
        sum += e;
    return sum;
}

template <typename GradView>
Float host_sum_grad_l2(const GradView& grads)
{
    auto values = grads.values();
    if(values.size() == 0)
        return static_cast<Float>(0.0);
    std::vector<Vector3> host(values.size());
    values.copy_to(host.data());
    Float sum = static_cast<Float>(0.0);
    for(const auto& g : host)
        sum += g.norm();
    return sum;
}
}  // namespace


// CoreX miscompiles complex Launch().apply() lambdas (gradient/Hessian
// computations produce garbage).  Explicit __global__ kernels bypass
// the lambda code-generation path and compile correctly.

__global__ void kernel_PP_contact_assemble(
    int pp_count, bool gradient_only,
    muda::CDense2D<ContactCoeff> table,
    muda::CDense1D<int> contact_ids,
    muda::CDense1D<Vector3> Ps,
    muda::CDense1D<Float> thicknesses,
    muda::CDense1D<Float> d_hats,
    Float dt,
    muda::CDense1D<Vector2i> PPs,
    muda::DoubletVectorViewer<Float, 3> PP_Gs,
    muda::TripletMatrixViewer<Float, 3> PP_Hs,
    CorexContactSpdStats* spd_stats,
    Float spd_ratio_trigger,
    Float pp_diag_scale)
{
    using namespace sym::codim_ipc_simplex_contact;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= pp_count) return;

    Vector2i PP = PPs(i);
    Vector2i cids = {contact_ids(PP[0]), contact_ids(PP[1])};
    Float kt2 = PP_kappa(table, cids) * dt * dt;

    const auto& P0 = Ps(PP[0]);
    const auto& P1 = Ps(PP[1]);

    Float thickness = PP_thickness(thicknesses(PP(0)), thicknesses(PP(1)));
    Float d_hat     = PP_d_hat(d_hats(PP(0)), d_hats(PP(1)));

    Vector2i flag = distance::point_point_distance_flag(P0, P1);

    Vector6 G;
    if(gradient_only)
    {
        PP_barrier_gradient(G, flag, kt2, d_hat, thickness, P0, P1);
        DoubletVectorAssembler DVA{PP_Gs};
        DVA.segment<2>(i * 2).write(PP, G);
    }
    else
    {
        Matrix6x6 H;
        corex_record_contact_inputs(spd_stats, CorexSpdPP, kt2, d_hat, thickness);
        Matrix6x6 H_before;
#if UIPC_ENABLE_GIPC_NATIVE_CONTACT
        PP_barrier_gradient_gipc_native_hessian(
            G, H, flag, kt2, d_hat, thickness, P0, P1);
        H_before = H;
#else
        PP_barrier_gradient_hessian(G, H, flag, kt2, d_hat, thickness, P0, P1);
        H_before = H;
        make_spd(H);
        corex_apply_spd_conditioning(H_before, H, spd_ratio_trigger, pp_diag_scale);
#endif
        corex_record_spd_projection(spd_stats, CorexSpdPP, H_before, H);
        DoubletVectorAssembler DVA{PP_Gs};
        DVA.segment<2>(i * 2).write(PP, G);
        TripletMatrixAssembler TMA{PP_Hs};
        TMA.half_block<2>(i * SimplexNormalContact::PPHalfHessianSize).write(PP, H);
    }
}

__global__ void kernel_PE_contact_assemble(
    int pe_count, bool gradient_only,
    muda::CDense2D<ContactCoeff> table,
    muda::CDense1D<int> contact_ids,
    muda::CDense1D<IndexT> body_ids,
    muda::CDense1D<Vector3> Ps,
    muda::CDense1D<Float> thicknesses,
    muda::CDense1D<Float> d_hats,
    Float dt,
    muda::CDense1D<Vector3i> PEs,
    muda::DoubletVectorViewer<Float, 3> PE_Gs,
    muda::TripletMatrixViewer<Float, 3> PE_Hs,
    CorexContactSpdStats* spd_stats,
    Float pe_diag_reg,
    Float pe_kappa_scale,
    Float spd_ratio_trigger,
    Float pe_diag_scale)
{
    using namespace sym::codim_ipc_simplex_contact;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= pe_count) return;

    Vector3i PE = PEs(i);
    Vector3i cids = {contact_ids(PE[0]), contact_ids(PE[1]), contact_ids(PE[2])};
    Vector3i bids = {body_ids(PE[0]), body_ids(PE[1]), body_ids(PE[2])};
    Float kt2 = PE_kappa(table, cids) * dt * dt;
    kt2 *= pe_kappa_scale;

    const auto& P  = Ps(PE[0]);
    const auto& E0 = Ps(PE[1]);
    const auto& E1 = Ps(PE[2]);

    Float thickness = PE_thickness(thicknesses(PE(0)), thicknesses(PE(1)), thicknesses(PE(2)));
    Float d_hat     = PE_d_hat(d_hats(PE(0)), d_hats(PE(1)), d_hats(PE(2)));

    Vector3i flag = distance::point_edge_distance_flag(P, E0, E1);

    Vector9 G;
    if(gradient_only)
    {
        PE_barrier_gradient(G, flag, kt2, d_hat, thickness, P, E0, E1);
        DoubletVectorAssembler DVA{PE_Gs};
        DVA.segment<3>(i * 3).write(PE, G);
    }
    else
    {
        Matrix9x9 H;
        corex_record_contact_inputs(spd_stats, CorexSpdPE, kt2, d_hat, thickness);
        Matrix9x9 H_before;
#if UIPC_ENABLE_GIPC_NATIVE_CONTACT
        PE_barrier_gradient_gipc_native_hessian(
            G, H, flag, kt2, d_hat, thickness, P, E0, E1);
        if(pe_diag_reg > static_cast<Float>(0))
            for(int d = 0; d < 9; ++d)
                H(d, d) += pe_diag_reg;
        H_before = H;
#else
        PE_barrier_gradient_hessian(G, H, flag, kt2, d_hat, thickness, P, E0, E1);
        if(pe_diag_reg > static_cast<Float>(0))
            for(int d = 0; d < 9; ++d)
                H(d, d) += pe_diag_reg;
        H_before = H;
        make_spd(H);
        corex_apply_spd_conditioning(H_before, H, spd_ratio_trigger, pe_diag_scale);
#endif
        corex_record_spd_projection(spd_stats, CorexSpdPE, H_before, H);
        corex_record_pe_outlier(spd_stats, i, PE, bids, H_before, H);
        DoubletVectorAssembler DVA{PE_Gs};
        DVA.segment<3>(i * 3).write(PE, G);
        TripletMatrixAssembler TMA{PE_Hs};
        TMA.half_block<3>(i * SimplexNormalContact::PEHalfHessianSize).write(PE, H);
    }
}

__global__ void kernel_EE_contact_assemble(
    int ee_count, bool gradient_only,
    muda::CDense2D<ContactCoeff> table,
    muda::CDense1D<int> contact_ids,
    muda::CDense1D<Vector3> Ps,
    muda::CDense1D<Vector3> rest_Ps,
    muda::CDense1D<Float> thicknesses,
    muda::CDense1D<Float> d_hats,
    Float dt,
    muda::CDense1D<Vector4i> EEs,
    muda::DoubletVectorViewer<Float, 3> EE_Gs,
    muda::TripletMatrixViewer<Float, 3> EE_Hs,
    CorexContactSpdStats* spd_stats,
    Float spd_ratio_trigger,
    Float ee_diag_scale)
{
    using namespace sym::codim_ipc_simplex_contact;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= ee_count) return;

    Vector4i EE = EEs(i);
    Vector4i cids = {contact_ids(EE[0]), contact_ids(EE[1]),
                     contact_ids(EE[2]), contact_ids(EE[3])};
    Float kt2 = EE_kappa(table, cids) * dt * dt;

    const auto& Ea0 = Ps(EE[0]);
    const auto& Ea1 = Ps(EE[1]);
    const auto& Eb0 = Ps(EE[2]);
    const auto& Eb1 = Ps(EE[3]);

    const auto& t0_Ea0 = rest_Ps(EE[0]);
    const auto& t0_Ea1 = rest_Ps(EE[1]);
    const auto& t0_Eb0 = rest_Ps(EE[2]);
    const auto& t0_Eb1 = rest_Ps(EE[3]);

    Float thickness = EE_thickness(thicknesses(EE(0)), thicknesses(EE(1)),
                                   thicknesses(EE(2)), thicknesses(EE(3)));
    Float d_hat     = EE_d_hat(d_hats(EE(0)), d_hats(EE(1)),
                               d_hats(EE(2)), d_hats(EE(3)));

    Vector4i flag = distance::edge_edge_distance_flag(Ea0, Ea1, Eb0, Eb1);

    Vector12 G;
    if(gradient_only)
    {
        mollified_EE_barrier_gradient(G, flag, kt2, d_hat, thickness,
                                      t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1,
                                      Ea0, Ea1, Eb0, Eb1);
        DoubletVectorAssembler DVA{EE_Gs};
        DVA.segment<4>(i * 4).write(EE, G);
    }
    else
    {
        Matrix12x12 H;
        corex_record_contact_inputs(spd_stats, CorexSpdEE, kt2, d_hat, thickness);
        Matrix12x12 H_before;
#if UIPC_ENABLE_GIPC_NATIVE_CONTACT
        mollified_EE_barrier_gradient_gipc_native_hessian(
            G, H, flag, kt2, d_hat, thickness,
            t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1,
            Ea0, Ea1, Eb0, Eb1);
        H_before = H;
#else
        mollified_EE_barrier_gradient_hessian(G, H, flag, kt2, d_hat, thickness,
                                               t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1,
                                               Ea0, Ea1, Eb0, Eb1);
        H_before = H;
        make_spd(H);
        corex_apply_spd_conditioning(H_before, H, spd_ratio_trigger, ee_diag_scale);
#endif
        corex_record_spd_projection(spd_stats, CorexSpdEE, H_before, H);
        DoubletVectorAssembler DVA{EE_Gs};
        DVA.segment<4>(i * 4).write(EE, G);
        TripletMatrixAssembler TMA{EE_Hs};
        TMA.half_block<4>(i * SimplexNormalContact::EEHalfHessianSize).write(EE, H);
    }
}

__global__ void kernel_PT_contact_assemble(
    int pt_count, bool gradient_only,
    muda::CDense2D<ContactCoeff> table,
    muda::CDense1D<int> contact_ids,
    muda::CDense1D<Vector3> Ps,
    muda::CDense1D<Float> thicknesses,
    muda::CDense1D<Float> d_hats,
    Float dt,
    muda::CDense1D<Vector4i> PTs,
    muda::DoubletVectorViewer<Float, 3> PT_Gs,
    muda::TripletMatrixViewer<Float, 3> PT_Hs,
    CorexContactSpdStats* spd_stats,
    Float spd_ratio_trigger,
    Float pt_diag_scale)
{
    using namespace sym::codim_ipc_simplex_contact;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= pt_count) return;

    Vector4i PT = PTs(i);
    Vector4i cids = {contact_ids(PT[0]), contact_ids(PT[1]),
                     contact_ids(PT[2]), contact_ids(PT[3])};
    Float kt2 = PT_kappa(table, cids) * dt * dt;

    const auto& P  = Ps(PT[0]);
    const auto& T0 = Ps(PT[1]);
    const auto& T1 = Ps(PT[2]);
    const auto& T2 = Ps(PT[3]);

    Float thickness = PT_thickness(thicknesses(PT(0)), thicknesses(PT(1)),
                                   thicknesses(PT(2)), thicknesses(PT(3)));
    Float d_hat     = PT_d_hat(d_hats(PT(0)), d_hats(PT(1)),
                               d_hats(PT(2)), d_hats(PT(3)));

    Vector4i flag = distance::point_triangle_distance_flag(P, T0, T1, T2);

    Vector12 G;
    if(gradient_only)
    {
        PT_barrier_gradient(G, flag, kt2, d_hat, thickness, P, T0, T1, T2);
        DoubletVectorAssembler DVA{PT_Gs};
        DVA.segment<4>(i * 4).write(PT, G);
    }
    else
    {
        Matrix12x12 H;
        corex_record_contact_inputs(spd_stats, CorexSpdPT, kt2, d_hat, thickness);
        Matrix12x12 H_before;
#if UIPC_ENABLE_GIPC_NATIVE_CONTACT
        PT_barrier_gradient_gipc_native_hessian(
            G, H, flag, kt2, d_hat, thickness, P, T0, T1, T2);
        H_before = H;
#else
        PT_barrier_gradient_hessian(G, H, flag, kt2, d_hat, thickness,
                                    P, T0, T1, T2);
        H_before = H;
        make_spd(H);
        corex_apply_spd_conditioning(H_before, H, spd_ratio_trigger, pt_diag_scale);
#endif
        corex_record_spd_projection(spd_stats, CorexSpdPT, H_before, H);
        DoubletVectorAssembler DVA{PT_Gs};
        DVA.segment<4>(i * 4).write(PT, G);
        TripletMatrixAssembler TMA{PT_Hs};
        TMA.half_block<4>(i * SimplexNormalContact::PTHalfHessianSize).write(PT, H);
    }
}

__global__ void kernel_PT_energy(
    int count,
    muda::CDense2D<ContactCoeff> table,
    muda::CDense1D<int> contact_ids,
    muda::CDense1D<Vector4i> PTs,
    muda::Dense1D<Float> Es,
    muda::CDense1D<Vector3> Ps,
    muda::CDense1D<Float> thicknesses,
    muda::CDense1D<Float> d_hats,
    Float dt)
{
    using namespace sym::codim_ipc_simplex_contact;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count) return;
    Vector4i PT = PTs(i);
    Vector4i cids = {contact_ids(PT[0]), contact_ids(PT[1]),
                     contact_ids(PT[2]), contact_ids(PT[3])};
    Float kt2 = PT_kappa(table, cids) * dt * dt;
    const auto& P  = Ps(PT[0]);
    const auto& T0 = Ps(PT[1]);
    const auto& T1 = Ps(PT[2]);
    const auto& T2 = Ps(PT[3]);
    Float thickness = PT_thickness(thicknesses(PT(0)), thicknesses(PT(1)),
                                   thicknesses(PT(2)), thicknesses(PT(3)));
    Float d_hat = PT_d_hat(d_hats(PT(0)), d_hats(PT(1)),
                           d_hats(PT(2)), d_hats(PT(3)));
    Vector4i flag = distance::point_triangle_distance_flag(P, T0, T1, T2);
    Es(i) = PT_barrier_energy(flag, kt2, d_hat, thickness, P, T0, T1, T2);
}

__global__ void kernel_EE_energy(
    int count,
    muda::CDense2D<ContactCoeff> table,
    muda::CDense1D<int> contact_ids,
    muda::CDense1D<Vector4i> EEs,
    muda::Dense1D<Float> Es,
    muda::CDense1D<Vector3> Ps,
    muda::CDense1D<Vector3> rest_Ps,
    muda::CDense1D<Float> thicknesses,
    muda::CDense1D<Float> d_hats,
    Float dt)
{
    using namespace sym::codim_ipc_simplex_contact;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count) return;
    Vector4i EE = EEs(i);
    Vector4i cids = {contact_ids(EE[0]), contact_ids(EE[1]),
                     contact_ids(EE[2]), contact_ids(EE[3])};
    Float kt2 = EE_kappa(table, cids) * dt * dt;
    const auto& Ea0 = Ps(EE[0]);
    const auto& Ea1 = Ps(EE[1]);
    const auto& Eb0 = Ps(EE[2]);
    const auto& Eb1 = Ps(EE[3]);
    const auto& t0_Ea0 = rest_Ps(EE[0]);
    const auto& t0_Ea1 = rest_Ps(EE[1]);
    const auto& t0_Eb0 = rest_Ps(EE[2]);
    const auto& t0_Eb1 = rest_Ps(EE[3]);
    Float thickness = EE_thickness(thicknesses(EE(0)), thicknesses(EE(1)),
                                   thicknesses(EE(2)), thicknesses(EE(3)));
    Float d_hat = EE_d_hat(d_hats(EE(0)), d_hats(EE(1)),
                           d_hats(EE(2)), d_hats(EE(3)));
    Vector4i flag = distance::edge_edge_distance_flag(Ea0, Ea1, Eb0, Eb1);
    Es(i) = mollified_EE_barrier_energy(flag, kt2, d_hat, thickness,
                                        t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1,
                                        Ea0, Ea1, Eb0, Eb1);
}

__global__ void kernel_PE_energy(
    int count,
    muda::CDense2D<ContactCoeff> table,
    muda::CDense1D<int> contact_ids,
    muda::CDense1D<Vector3i> PEs,
    muda::Dense1D<Float> Es,
    muda::CDense1D<Vector3> Ps,
    muda::CDense1D<Float> thicknesses,
    muda::CDense1D<Float> d_hats,
    Float dt)
{
    using namespace sym::codim_ipc_simplex_contact;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count) return;
    Vector3i PE = PEs(i);
    Vector3i cids = {contact_ids(PE[0]), contact_ids(PE[1]), contact_ids(PE[2])};
    Float kt2 = PE_kappa(table, cids) * dt * dt;
    const auto& P  = Ps(PE[0]);
    const auto& E0 = Ps(PE[1]);
    const auto& E1 = Ps(PE[2]);
    Float thickness = PE_thickness(thicknesses(PE(0)), thicknesses(PE(1)),
                                   thicknesses(PE(2)));
    Float d_hat = PE_d_hat(d_hats(PE(0)), d_hats(PE(1)), d_hats(PE(2)));
    Vector3i flag = distance::point_edge_distance_flag(P, E0, E1);
    Es(i) = PE_barrier_energy(flag, kt2, d_hat, thickness, P, E0, E1);
}

__global__ void kernel_PP_energy(
    int count,
    muda::CDense2D<ContactCoeff> table,
    muda::CDense1D<int> contact_ids,
    muda::CDense1D<Vector2i> PPs,
    muda::Dense1D<Float> Es,
    muda::CDense1D<Vector3> Ps,
    muda::CDense1D<Float> thicknesses,
    muda::CDense1D<Float> d_hats,
    Float dt)
{
    using namespace sym::codim_ipc_simplex_contact;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count) return;
    Vector2i PP = PPs(i);
    Vector2i cids = {contact_ids(PP[0]), contact_ids(PP[1])};
    Float kt2 = PP_kappa(table, cids) * dt * dt;
    const auto& P0 = Ps(PP[0]);
    const auto& P1 = Ps(PP[1]);
    Float thickness = PP_thickness(thicknesses(PP(0)), thicknesses(PP(1)));
    Float d_hat = PP_d_hat(d_hats(PP(0)), d_hats(PP(1)));
    Vector2i flag = distance::point_point_distance_flag(P0, P1);
    Es(i) = PP_barrier_energy(flag, kt2, d_hat, thickness, P0, P1);
}

class IPCSimplexNormalContact final : public SimplexNormalContact
{
  public:
    using SimplexNormalContact::SimplexNormalContact;

    virtual void do_build(BuildInfo& info) override
    {
        require<IPCPipelineFlag>();
    }

    virtual void do_compute_energy(EnergyInfo& info) override
    {
        using namespace muda;
        using namespace sym::codim_ipc_simplex_contact;

        constexpr int kBlk = 256;
        auto grid = [](int n) { return (n + kBlk - 1) / kBlk; };

        auto PT_count = (IndexT)info.PTs().size();
        if(PT_count > 0)
            kernel_PT_energy<<<grid(PT_count), kBlk>>>(
                PT_count,
                info.contact_tabular().viewer(),
                info.contact_element_ids().viewer(),
                info.PTs().viewer(),
                info.PT_energies().viewer(),
                info.positions().viewer(),
                info.thicknesses().viewer(),
                info.d_hats().viewer(),
                info.dt());

        auto EE_count = (IndexT)info.EEs().size();
        if(EE_count > 0)
            kernel_EE_energy<<<grid(EE_count), kBlk>>>(
                EE_count,
                info.contact_tabular().viewer(),
                info.contact_element_ids().viewer(),
                info.EEs().viewer(),
                info.EE_energies().viewer(),
                info.positions().viewer(),
                info.rest_positions().viewer(),
                info.thicknesses().viewer(),
                info.d_hats().viewer(),
                info.dt());

        auto PE_count = (IndexT)info.PEs().size();
        if(PE_count > 0)
            kernel_PE_energy<<<grid(PE_count), kBlk>>>(
                PE_count,
                info.contact_tabular().viewer(),
                info.contact_element_ids().viewer(),
                info.PEs().viewer(),
                info.PE_energies().viewer(),
                info.positions().viewer(),
                info.thicknesses().viewer(),
                info.d_hats().viewer(),
                info.dt());

        auto PP_count = (IndexT)info.PPs().size();
        if(PP_count > 0)
            kernel_PP_energy<<<grid(PP_count), kBlk>>>(
                PP_count,
                info.contact_tabular().viewer(),
                info.contact_element_ids().viewer(),
                info.PPs().viewer(),
                info.PP_energies().viewer(),
                info.positions().viewer(),
                info.thicknesses().viewer(),
                info.d_hats().viewer(),
                info.dt());
        if(trace_contact_type_energy_enabled())
        {
            const Float pt = host_sum_energy(info.PT_energies());
            const Float ee = host_sum_energy(info.EE_energies());
            const Float pe = host_sum_energy(info.PE_energies());
            const Float pp = host_sum_energy(info.PP_energies());
            const Float total = pt + ee + pe + pp;
            const Float inv_total =
                total > static_cast<Float>(0.0) ? static_cast<Float>(1.0) / total
                                                : static_cast<Float>(0.0);
            spdlog::info("[corex_trace][simplex_normal_energy_mix] "
                         "counts PT={} EE={} PE={} PP={}, "
                         "energy PT={:.9g} ({:.3f}%) EE={:.9g} ({:.3f}%) "
                         "PE={:.9g} ({:.3f}%) PP={:.9g} ({:.3f}%) total={:.9g}",
                         info.PTs().size(),
                         info.EEs().size(),
                         info.PEs().size(),
                         info.PPs().size(),
                         pt,
                         static_cast<double>(pt * inv_total * 100.0),
                         ee,
                         static_cast<double>(ee * inv_total * 100.0),
                         pe,
                         static_cast<double>(pe * inv_total * 100.0),
                         pp,
                         static_cast<double>(pp * inv_total * 100.0),
                         total);
        }
    }

    virtual void do_assemble(ContactInfo& info) override
    {
        using namespace muda;
        using namespace sym::codim_ipc_simplex_contact;

        constexpr int kBlk = 256;
        auto grid = [](int n) { return (n + 255) / 256; };

        auto pp_count = (IndexT)info.PPs().size();
        auto pe_count = (IndexT)info.PEs().size();
        auto ee_count = (IndexT)info.EEs().size();
        auto pt_count = (IndexT)info.PTs().size();
        auto total    = pt_count + ee_count + pe_count + pp_count;
        constexpr Float pe_diag_reg = static_cast<Float>(0);
        constexpr Float pe_kappa_scale = static_cast<Float>(1);
        constexpr Float spd_ratio_trigger = static_cast<Float>(0.25);
        constexpr Float pp_diag_scale = static_cast<Float>(0);
        constexpr Float pe_cond_diag_scale = static_cast<Float>(0);
        constexpr Float ee_diag_scale = static_cast<Float>(0);
        constexpr Float pt_diag_scale = static_cast<Float>(0);

        // CoreX: use explicit __global__ kernels (lambdas miscompile)
        const bool spd_diag_enabled =
            corex_contact_spd_diag_enabled() && !info.gradient_only();
        const bool pe_outlier_diag_enabled =
            corex_pe_outlier_diag_enabled() && !info.gradient_only();
        CorexContactSpdStats* spd_stats = nullptr;
        if(corex_contact_spd_diag_enabled() || pe_outlier_diag_enabled)
            spdlog::info("[corex_spd_contact_begin] total={} PT={} EE={} PE={} PP={} gradient_only={} enabled={}",
                         total,
                         pt_count,
                         ee_count,
                         pe_count,
                         pp_count,
                         info.gradient_only() ? 1 : 0,
                         spd_diag_enabled ? 1 : 0);
        if(spd_diag_enabled || pe_outlier_diag_enabled)
        {
            cudaMalloc(reinterpret_cast<void**>(&spd_stats), sizeof(CorexContactSpdStats));
            cudaMemset(spd_stats, 0, sizeof(CorexContactSpdStats));
            if(corex_contact_input_diag_enabled())
            {
                unsigned int enabled = 1u;
                cudaMemcpy(&spd_stats->input_diag_enabled,
                           &enabled,
                           sizeof(enabled),
                           cudaMemcpyHostToDevice);
            }
            if(pe_outlier_diag_enabled)
            {
                unsigned int enabled = 1u;
                Float ratio_trigger = corex_env_float("UIPC_COREX_PE_OUTLIER_RATIO_TRIGGER",
                                                      static_cast<Float>(0.5));
                cudaMemcpy(&spd_stats->pe_outlier_diag_enabled,
                           &enabled,
                           sizeof(enabled),
                           cudaMemcpyHostToDevice);
                cudaMemcpy(&spd_stats->pe_outlier_ratio_trigger,
                           &ratio_trigger,
                           sizeof(ratio_trigger),
                           cudaMemcpyHostToDevice);
            }
        }

        if(pp_count > 0)
            kernel_PP_contact_assemble<<<grid(pp_count), kBlk>>>(
                pp_count, info.gradient_only(),
                info.contact_tabular().viewer(),
                info.contact_element_ids().viewer(),
                info.positions().viewer(),
                info.thicknesses().viewer(),
                info.d_hats().viewer(),
                info.dt(),
                info.PPs().viewer(),
                info.PP_gradients().viewer(),
                info.PP_hessians().viewer(),
                spd_stats,
                spd_ratio_trigger,
                pp_diag_scale);

        if(pe_count > 0)
            kernel_PE_contact_assemble<<<grid(pe_count), kBlk>>>(
                pe_count, info.gradient_only(),
                info.contact_tabular().viewer(),
                info.contact_element_ids().viewer(),
                info.body_ids().viewer(),
                info.positions().viewer(),
                info.thicknesses().viewer(),
                info.d_hats().viewer(),
                info.dt(),
                info.PEs().viewer(),
                info.PE_gradients().viewer(),
                info.PE_hessians().viewer(),
                spd_stats,
                pe_diag_reg,
                pe_kappa_scale,
                spd_ratio_trigger,
                pe_cond_diag_scale);

        if(ee_count > 0)
            kernel_EE_contact_assemble<<<grid(ee_count), kBlk>>>(
                ee_count, info.gradient_only(),
                info.contact_tabular().viewer(),
                info.contact_element_ids().viewer(),
                info.positions().viewer(),
                info.rest_positions().viewer(),
                info.thicknesses().viewer(),
                info.d_hats().viewer(),
                info.dt(),
                info.EEs().viewer(),
                info.EE_gradients().viewer(),
                info.EE_hessians().viewer(),
                spd_stats,
                spd_ratio_trigger,
                ee_diag_scale);

        if(pt_count > 0)
            kernel_PT_contact_assemble<<<grid(pt_count), kBlk>>>(
                pt_count, info.gradient_only(),
                info.contact_tabular().viewer(),
                info.contact_element_ids().viewer(),
                info.positions().viewer(),
                info.thicknesses().viewer(),
                info.d_hats().viewer(),
                info.dt(),
                info.PTs().viewer(),
                info.PT_gradients().viewer(),
                info.PT_hessians().viewer(),
                spd_stats,
                spd_ratio_trigger,
                pt_diag_scale);

        if(spd_diag_enabled || pe_outlier_diag_enabled)
        {
            cudaDeviceSynchronize();
            CorexContactSpdStats h_stats{};
            cudaMemcpy(&h_stats, spd_stats, sizeof(CorexContactSpdStats), cudaMemcpyDeviceToHost);
            const char* names[4] = {"PT", "EE", "PE", "PP"};
            if(spd_diag_enabled)
            {
                for(int t = 0; t < 4; ++t)
                {
                    auto count = h_stats.count[t];
                    double inv_count = count > 0 ? 1.0 / static_cast<double>(count) : 0.0;
                    spdlog::info("[corex_spd_contact] type={} count={} "
                                 "projected={} projected_rate={:.3f} "
                                 "gershgorin_lower_avg={:.9g} correction_avg={:.9g} "
                                 "correction_max={:.9g} diag_abs_avg={:.9g} "
                                 "correction_diag_ratio_avg={:.9g} correction_diag_ratio_max={:.9g}",
                                 names[t],
                                 count,
                                 h_stats.projected[t],
                                 100.0 * static_cast<double>(h_stats.projected[t]) * inv_count,
                                 static_cast<double>(h_stats.lower_sum[t]) * inv_count,
                                 static_cast<double>(h_stats.correction_sum[t]) * inv_count,
                                 static_cast<double>(h_stats.max_correction[t]),
                                 static_cast<double>(h_stats.diag_abs_sum[t]) * inv_count,
                                 static_cast<double>(h_stats.correction_diag_ratio_sum[t]) * inv_count,
                                 static_cast<double>(h_stats.max_correction_diag_ratio[t]));
                    if(corex_contact_input_diag_enabled())
                    {
                        auto input_count = h_stats.input_count[t];
                        double inv_input =
                            input_count > 0 ? 1.0 / static_cast<double>(input_count) : 0.0;
                        spdlog::info("[corex_contact_input] type={} count={} "
                                     "kt2_avg={:.9g} kt2_max_abs={:.9g} "
                                     "d_hat_avg={:.9g} d_hat_max_abs={:.9g} "
                                     "thickness_avg={:.9g} thickness_max_abs={:.9g}",
                                     names[t],
                                     input_count,
                                     static_cast<double>(h_stats.kt2_sum[t]) * inv_input,
                                     static_cast<double>(h_stats.kt2_max[t]),
                                     static_cast<double>(h_stats.d_hat_sum[t]) * inv_input,
                                     static_cast<double>(h_stats.d_hat_max[t]),
                                     static_cast<double>(h_stats.thickness_sum[t]) * inv_input,
                                     static_cast<double>(h_stats.thickness_max[t]));
                    }
                }
            }
            if(pe_outlier_diag_enabled)
                spdlog::info("[corex_pe_outlier] count_ge_trigger={} ratio_trigger={:.9g} "
                             "max_ratio={:.9g} correction={:.9g} diag_abs={:.9g} lower={:.9g} "
                             "contact_i={} PE=({},{},{}) body=({},{},{})",
                             h_stats.pe_outlier_count,
                             static_cast<double>(h_stats.pe_outlier_ratio_trigger),
                             static_cast<double>(h_stats.pe_outlier_ratio_max),
                             static_cast<double>(h_stats.pe_outlier_corr_max),
                             static_cast<double>(h_stats.pe_outlier_diag_abs),
                             static_cast<double>(h_stats.pe_outlier_lower),
                             h_stats.pe_outlier_i,
                             h_stats.pe_outlier_v0,
                             h_stats.pe_outlier_v1,
                             h_stats.pe_outlier_v2,
                             h_stats.pe_outlier_b0,
                             h_stats.pe_outlier_b1,
                             h_stats.pe_outlier_b2);
            cudaFree(spd_stats);
        }

        if(trace_contact_type_gradient_enabled())
        {
            const Float pt = host_sum_grad_l2(info.PT_gradients());
            const Float ee = host_sum_grad_l2(info.EE_gradients());
            const Float pe = host_sum_grad_l2(info.PE_gradients());
            const Float pp = host_sum_grad_l2(info.PP_gradients());
            const Float total = pt + ee + pe + pp;
            const Float inv_total =
                total > static_cast<Float>(0.0) ? static_cast<Float>(1.0) / total
                                                : static_cast<Float>(0.0);
            spdlog::info("[corex_trace][simplex_normal_grad_mix] "
                         "doublets PT={} EE={} PE={} PP={}, "
                         "l2sum PT={:.9g} ({:.3f}%) EE={:.9g} ({:.3f}%) "
                         "PE={:.9g} ({:.3f}%) PP={:.9g} ({:.3f}%) total={:.9g}",
                         info.PT_gradients().doublet_count(),
                         info.EE_gradients().doublet_count(),
                         info.PE_gradients().doublet_count(),
                         info.PP_gradients().doublet_count(),
                         pt,
                         static_cast<double>(pt * inv_total * 100.0),
                         ee,
                         static_cast<double>(ee * inv_total * 100.0),
                         pe,
                         static_cast<double>(pe * inv_total * 100.0),
                         pp,
                         static_cast<double>(pp * inv_total * 100.0),
                         total);
        }

        if(trace_barrier_dbdd_enabled())
        {
            using namespace sym::codim_ipc_contact;
            auto pe_count = (IndexT)info.PEs().size();
            auto ee_count = (IndexT)info.EEs().size();
            auto pp_count = (IndexT)info.PPs().size();
            auto pt_count = (IndexT)info.PTs().size();
            auto total_pairs = pe_count + ee_count + pp_count + pt_count;
            if(total_pairs > 0)
            {
                auto n_pos = info.positions().size();
                std::vector<Vector3> h_pos(n_pos);
                info.positions().copy_to(h_pos.data());
                std::vector<Float> h_thick(n_pos), h_dhats(n_pos);
                info.thicknesses().copy_to(h_thick.data());
                info.d_hats().copy_to(h_dhats.data());

                Float dBdD_min = std::numeric_limits<Float>::max();
                Float dBdD_max = -std::numeric_limits<Float>::max();
                int   sample_count = 0;

                auto sample_dBdD = [&](Float D, Float d_hat, Float thickness)
                {
                    const Float kappa_unit = static_cast<Float>(1.0);
                    Float dBdD;
                    dKappaBarrierdD(dBdD, kappa_unit, D, d_hat, thickness);
                    if(dBdD < dBdD_min) dBdD_min = dBdD;
                    if(dBdD > dBdD_max) dBdD_max = dBdD;
                    ++sample_count;
                    return dBdD;
                };

                if(pe_count > 0)
                {
                    std::vector<Vector3i> h_pes(pe_count);
                    info.PEs().copy_to(h_pes.data());
                    for(int i = 0; i < (int)pe_count && i < 4; ++i)
                    {
                        auto PE = h_pes[i];
                        Float thick = PE_thickness(h_thick[PE(0)], h_thick[PE(1)], h_thick[PE(2)]);
                        Float dhat  = PE_d_hat(h_dhats[PE(0)], h_dhats[PE(1)], h_dhats[PE(2)]);
                        auto flag = distance::point_edge_distance_flag(h_pos[PE(0)], h_pos[PE(1)], h_pos[PE(2)]);
                        Float D;
                        distance::point_edge_distance2(flag, h_pos[PE(0)], h_pos[PE(1)], h_pos[PE(2)], D);
                        Float V = dhat * dhat + Float(2) * dhat * thick;
                        Float dBdD = sample_dBdD(D, dhat, thick);
                        spdlog::info("[corex_trace][barrier_dBdD] PE[{}] D={:.6g} V={:.6g} D/V={:.4f} dBdD(k=1)={:.6g}",
                                     i, (double)D, (double)V, (double)(V > 0 ? D/V : 0), (double)dBdD);
                    }
                }

                if(ee_count > 0)
                {
                    std::vector<Vector4i> h_ees(ee_count);
                    info.EEs().copy_to(h_ees.data());
                    for(int i = 0; i < (int)ee_count && i < 4; ++i)
                    {
                        auto EE = h_ees[i];
                        Float thick = EE_thickness(h_thick[EE(0)], h_thick[EE(1)], h_thick[EE(2)], h_thick[EE(3)]);
                        Float dhat  = EE_d_hat(h_dhats[EE(0)], h_dhats[EE(1)], h_dhats[EE(2)], h_dhats[EE(3)]);
                        auto flag = distance::edge_edge_distance_flag(h_pos[EE(0)], h_pos[EE(1)], h_pos[EE(2)], h_pos[EE(3)]);
                        Float D;
                        distance::edge_edge_distance2(flag, h_pos[EE(0)], h_pos[EE(1)], h_pos[EE(2)], h_pos[EE(3)], D);
                        Float V = dhat * dhat + Float(2) * dhat * thick;
                        Float dBdD = sample_dBdD(D, dhat, thick);
                        spdlog::info("[corex_trace][barrier_dBdD] EE[{}] D={:.6g} V={:.6g} D/V={:.4f} dBdD(k=1)={:.6g}",
                                     i, (double)D, (double)V, (double)(V > 0 ? D/V : 0), (double)dBdD);
                    }
                }

                spdlog::info("[corex_trace][barrier_dBdD_summary] samples={} dBdD_range(k=1)=[{:.6g}, {:.6g}] "
                             "PP={} PE={} EE={} PT={}",
                             sample_count, (double)dBdD_min, (double)dBdD_max,
                             pp_count, pe_count, ee_count, pt_count);
            }
        }
    }
};

REGISTER_SIM_SYSTEM(IPCSimplexNormalContact);
}  // namespace uipc::backend::cuda
#else
#include <contact_system/simplex_normal_contact.h>
#include <contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h>
#include <utils/distance/distance_flagged.h>
#include <utils/codim_thickness.h>
#include <kernel_cout.h>
#include <utils/matrix_assembler.h>
#include <utils/make_spd.h>
#include <utils/primitive_d_hat.h>
#include <pipeline/ipc_pipeline_flag.h>

namespace uipc::backend::cuda
{
namespace
{
bool corex_contact_spd_diag_enabled()
{
    const char* env = std::getenv("UIPC_COREX_CONTACT_SPD_DIAG");
    if(!env) return false;
    return env[0] != '\0' && env[0] != '0';
}

struct CorexContactSpdStats
{
    unsigned int count[4];
    unsigned int projected[4];
    Float lower_sum[4];
    Float diag_abs_sum[4];
    Float correction_sum[4];
    Float correction_diag_ratio_sum[4];
    Float max_correction[4];
    Float max_correction_diag_ratio[4];
};

enum CorexContactSpdType : int
{
    CorexSpdPT = 0,
    CorexSpdEE = 1,
    CorexSpdPE = 2,
    CorexSpdPP = 3,
};

__device__ inline Float corex_abs_float(Float v)
{
    return v < static_cast<Float>(0) ? -v : v;
}

__device__ inline void corex_atomic_max_float(Float* address, Float value)
{
#if defined(UIPC_FLOAT_SCALAR) && UIPC_FLOAT_SCALAR
    int* address_as_i = reinterpret_cast<int*>(address);
    int  old          = *address_as_i;
    int  assumed;
    do
    {
        assumed = old;
        Float old_v = __int_as_float(assumed);
        if(old_v >= value)
            break;
        old = atomicCAS(address_as_i, assumed, __float_as_int(value));
    } while(assumed != old);
#else
    unsigned long long int* address_as_i =
        reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_i;
    unsigned long long int assumed;
    do
    {
        assumed = old;
        Float old_v = __longlong_as_double(assumed);
        if(old_v >= value)
            break;
        old = atomicCAS(address_as_i, assumed, __double_as_longlong(value));
    } while(assumed != old);
#endif
}

template <int N>
__device__ Float corex_gershgorin_lower(const Matrix<Float, N, N>& H)
{
    Float lower_min = H(0, 0);
    for(int i = 0; i < N; ++i)
    {
        Float off_sum = static_cast<Float>(0);
        for(int j = 0; j < N; ++j)
            if(i != j)
                off_sum += corex_abs_float(H(i, j));
        Float lower = H(i, i) - off_sum;
        if(lower < lower_min)
            lower_min = lower;
    }
    return lower_min;
}

template <int N>
__device__ void corex_record_spd_projection(CorexContactSpdStats* stats,
                                            CorexContactSpdType   type,
                                            const Matrix<Float, N, N>& before,
                                            const Matrix<Float, N, N>& after)
{
    if(!stats)
        return;
    Float lower = corex_gershgorin_lower(before);
    Float diag_abs = static_cast<Float>(0);
    for(int i = 0; i < N; ++i)
        diag_abs += corex_abs_float(before(i, i));
    Float correction = static_cast<Float>(0);
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            correction += corex_abs_float(after(i, j) - before(i, j));
    Float correction_diag_ratio =
        diag_abs > static_cast<Float>(0) ? correction / diag_abs : static_cast<Float>(0);
    int t = static_cast<int>(type);
    atomicAdd(&stats->count[t], 1u);
    atomicAdd(&stats->lower_sum[t], lower);
    atomicAdd(&stats->diag_abs_sum[t], diag_abs);
    atomicAdd(&stats->correction_sum[t], correction);
    atomicAdd(&stats->correction_diag_ratio_sum[t], correction_diag_ratio);
    if(correction > static_cast<Float>(0))
        atomicAdd(&stats->projected[t], 1u);
    corex_atomic_max_float(&stats->max_correction[t], correction);
    corex_atomic_max_float(&stats->max_correction_diag_ratio[t], correction_diag_ratio);
}
}  // namespace

class IPCSimplexNormalContact final : public SimplexNormalContact
{
  public:
    using SimplexNormalContact::SimplexNormalContact;

    virtual void do_build(BuildInfo& info) override
    {
        require<IPCPipelineFlag>();
    }

    virtual void do_compute_energy(EnergyInfo& info) override
    {
        using namespace muda;
        using namespace sym::codim_ipc_simplex_contact;

        // Compute Point-Triangle energy
        auto PT_count = info.PTs().size();
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(PT_count,
                   [table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    PTs = info.PTs().viewer().name("PTs"),
                    Es  = info.PT_energies().viewer().name("Es"),
                    Ps  = info.positions().viewer().name("Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    d_hats = info.d_hats().viewer().name("d_hats"),
                    dt     = info.dt()] __device__(int i) mutable
                   {
                       Vector4i PT = PTs(i);

                       Vector4i cids = {contact_ids(PT[0]),
                                        contact_ids(PT[1]),
                                        contact_ids(PT[2]),
                                        contact_ids(PT[3])};
                       Float    kt2  = PT_kappa(table, cids) * dt * dt;

                       const auto& P  = Ps(PT[0]);
                       const auto& T0 = Ps(PT[1]);
                       const auto& T1 = Ps(PT[2]);
                       const auto& T2 = Ps(PT[3]);


                       Float thickness = PT_thickness(thicknesses(PT(0)),
                                                      thicknesses(PT(1)),
                                                      thicknesses(PT(2)),
                                                      thicknesses(PT(3)));

                       Float d_hat = PT_d_hat(
                           d_hats(PT(0)), d_hats(PT(1)), d_hats(PT(2)), d_hats(PT(3)));

                       Vector4i flag =
                           distance::point_triangle_distance_flag(P, T0, T1, T2);

                       if constexpr(RUNTIME_CHECK)
                       {
                           Float D;
                           distance::point_triangle_distance2(flag, P, T0, T1, T2, D);

                           Vector2 range = D_range(thickness, d_hat);

                           MUDA_ASSERT(is_active_D(range, D),
                                       "PT[%d,%d,%d,%d] d^2(%f) out of range, (%f,%f)",
                                       PT(0),
                                       PT(1),
                                       PT(2),
                                       PT(3),
                                       D,
                                       range(0),
                                       range(1));
                       }

                       Es(i) = PT_barrier_energy(flag, kt2, d_hat, thickness, P, T0, T1, T2);
                   });

        // Compute Edge-Edge energy
        auto EE_count = info.EEs().size();
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(EE_count,
                   [table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    EEs = info.EEs().viewer().name("EEs"),
                    Es  = info.EE_energies().viewer().name("Es"),
                    Ps  = info.positions().viewer().name("Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    rest_Ps = info.rest_positions().viewer().name("rest_Ps"),
                    d_hats  = info.d_hats().viewer().name("d_hats"),
                    dt      = info.dt()] __device__(int i) mutable
                   {
                       Vector4i EE = EEs(i);

                       Vector4i cids = {contact_ids(EE[0]),
                                        contact_ids(EE[1]),
                                        contact_ids(EE[2]),
                                        contact_ids(EE[3])};
                       Float    kt2  = EE_kappa(table, cids) * dt * dt;

                       const auto& E0 = Ps(EE[0]);
                       const auto& E1 = Ps(EE[1]);
                       const auto& E2 = Ps(EE[2]);
                       const auto& E3 = Ps(EE[3]);

                       const auto& t0_Ea0 = rest_Ps(EE[0]);
                       const auto& t0_Ea1 = rest_Ps(EE[1]);
                       const auto& t0_Eb0 = rest_Ps(EE[2]);
                       const auto& t0_Eb1 = rest_Ps(EE[3]);

                       Float thickness = EE_thickness(thicknesses(EE(0)),
                                                      thicknesses(EE(1)),
                                                      thicknesses(EE(2)),
                                                      thicknesses(EE(3)));

                       Float d_hat = EE_d_hat(
                           d_hats(EE(0)), d_hats(EE(1)), d_hats(EE(2)), d_hats(EE(3)));

                       Vector4i flag = distance::edge_edge_distance_flag(E0, E1, E2, E3);

                       if constexpr(RUNTIME_CHECK)
                       {
                           Float D;
                           distance::edge_edge_distance2(flag, E0, E1, E2, E3, D);
                           Vector2 range = D_range(thickness, d_hat);
                           MUDA_ASSERT(is_active_D(range, D),
                                       "EE[%d,%d,%d,%d] d^2(%f) out of range, (%f,%f)",
                                       EE(0),
                                       EE(1),
                                       EE(2),
                                       EE(3),
                                       D,
                                       range(0),
                                       range(1));
                       }


                       Es(i) = mollified_EE_barrier_energy(flag,
                                                           // coefficients
                                                           kt2,
                                                           d_hat,
                                                           thickness,
                                                           // positions
                                                           t0_Ea0,
                                                           t0_Ea1,
                                                           t0_Eb0,
                                                           t0_Eb1,
                                                           E0,
                                                           E1,
                                                           E2,
                                                           E3);
                   });

        // Compute Point-Edge energy
        auto PE_count = info.PEs().size();
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(PE_count,
                   [table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    PEs     = info.PEs().viewer().name("PEs"),
                    Es      = info.PE_energies().viewer().name("Es"),
                    Ps      = info.positions().viewer().name("Ps"),
                    rest_Ps = info.rest_positions().viewer().name("rest_Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    eps_v  = info.eps_velocity(),
                    d_hats = info.d_hats().viewer().name("d_hats"),
                    dt     = info.dt()] __device__(int i) mutable
                   {
                       Vector3i PE = PEs(i);

                       Vector3i cids = {contact_ids(PE[0]),
                                        contact_ids(PE[1]),
                                        contact_ids(PE[2])};
                       Float    kt2  = PE_kappa(table, cids) * dt * dt;

                       const auto& P  = Ps(PE[0]);
                       const auto& E0 = Ps(PE[1]);
                       const auto& E1 = Ps(PE[2]);

                       Float thickness = PE_thickness(thicknesses(PE(0)),
                                                      thicknesses(PE(1)),
                                                      thicknesses(PE(2)));

                       Float d_hat =
                           PE_d_hat(d_hats(PE(0)), d_hats(PE(1)), d_hats(PE(2)));

                       Vector3i flag = distance::point_edge_distance_flag(P, E0, E1);

                       if constexpr(RUNTIME_CHECK)
                       {
                           Float D;
                           distance::point_edge_distance2(flag, P, E0, E1, D);

                           Vector2 range = D_range(thickness, d_hat);

                           MUDA_ASSERT(is_active_D(range, D),
                                       "PE[%d,%d,%d] d^2(%f) out of range, (%f,%f)",
                                       PE(0),
                                       PE(1),
                                       PE(2),
                                       D,
                                       range(0),
                                       range(1));
                       }

                       Es(i) = PE_barrier_energy(flag, kt2, d_hat, thickness, P, E0, E1);
                   });

        // Compute Point-Point energy
        auto PP_count = info.PPs().size();
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(PP_count,
                   [table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    PPs     = info.PPs().viewer().name("PPs"),
                    Es      = info.PP_energies().viewer().name("Es"),
                    Ps      = info.positions().viewer().name("Ps"),
                    rest_Ps = info.rest_positions().viewer().name("rest_Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    d_hats = info.d_hats().viewer().name("d_hats"),
                    dt     = info.dt()] __device__(int i) mutable
                   {
                       Vector2i PP = PPs(i);

                       Vector2i cids = {contact_ids(PP[0]), contact_ids(PP[1])};
                       Float    kt2  = PP_kappa(table, cids) * dt * dt;

                       const auto& Pa = Ps(PP[0]);
                       const auto& Pb = Ps(PP[1]);

                       Float thickness =
                           PP_thickness(thicknesses(PP(0)), thicknesses(PP(1)));

                       Float d_hat = PP_d_hat(d_hats(PP(0)), d_hats(PP(1)));

                       Vector2i flag = distance::point_point_distance_flag(Pa, Pb);

                       if constexpr(RUNTIME_CHECK)
                       {
                           Float D;
                           distance::point_point_distance2(flag, Pa, Pb, D);

                           Vector2 range = D_range(thickness, d_hat);

                           MUDA_ASSERT(is_active_D(range, D),
                                       "PP[%d,%d] d^2(%f) out of range, (%f,%f)",
                                       PP(0),
                                       PP(1),
                                       D,
                                       range(0),
                                       range(1));
                       }

                       Es(i) = PP_barrier_energy(flag, kt2, d_hat, thickness, Pa, Pb);
                   });
    }

    virtual void do_assemble(ContactInfo& info) override
    {
        using namespace muda;
        using namespace sym::codim_ipc_simplex_contact;

        // Fused kernel: PT + EE + PE + PP in one launch using offset-based dispatch.
        // Reduces 4 kernel launches to 1, improving GPU occupancy by providing
        // more threads in a single launch.
        auto pt_count = (IndexT)info.PTs().size();
        auto ee_count = (IndexT)info.EEs().size();
        auto pe_count = (IndexT)info.PEs().size();
        auto pp_count = (IndexT)info.PPs().size();
        auto total    = pt_count + ee_count + pe_count + pp_count;

        if(total == 0)
            return;

        IndexT ee_offset = pt_count;
        IndexT pe_offset = ee_offset + ee_count;
        IndexT pp_offset = pe_offset + pe_count;

        const bool spd_diag_enabled =
            corex_contact_spd_diag_enabled() && !info.gradient_only();
        CorexContactSpdStats* spd_stats = nullptr;
        if(corex_contact_spd_diag_enabled())
            spdlog::info("[corex_spd_contact_begin] total={} PT={} EE={} PE={} PP={} gradient_only={} enabled={}",
                         total,
                         pt_count,
                         ee_count,
                         pe_count,
                         pp_count,
                         info.gradient_only() ? 1 : 0,
                         spd_diag_enabled ? 1 : 0);
        if(spd_diag_enabled)
        {
            cudaMalloc(reinterpret_cast<void**>(&spd_stats), sizeof(CorexContactSpdStats));
            cudaMemset(spd_stats, 0, sizeof(CorexContactSpdStats));
        }

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(
                total,
                [gradient_only = info.gradient_only(),
                 table = info.contact_tabular().viewer().name("contact_tabular"),
                 contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                 Ps          = info.positions().viewer().name("Ps"),
                 rest_Ps     = info.rest_positions().viewer().name("rest_Ps"),
                 thicknesses = info.thicknesses().viewer().name("thicknesses"),
                 d_hats      = info.d_hats().viewer().name("d_hats"),
                 dt          = info.dt(),
                 // PT
                 PTs   = info.PTs().viewer().name("PTs"),
                 PT_Gs = info.PT_gradients().viewer().name("PT_Gs"),
                 PT_Hs = info.PT_hessians().viewer().name("PT_Hs"),
                 // EE
                 EEs   = info.EEs().viewer().name("EEs"),
                 EE_Gs = info.EE_gradients().viewer().name("EE_Gs"),
                 EE_Hs = info.EE_hessians().viewer().name("EE_Hs"),
                 // PE
                 PEs   = info.PEs().viewer().name("PEs"),
                 PE_Gs = info.PE_gradients().viewer().name("PE_Gs"),
                 PE_Hs = info.PE_hessians().viewer().name("PE_Hs"),
                 // PP
                 PPs   = info.PPs().viewer().name("PPs"),
                 PP_Gs = info.PP_gradients().viewer().name("PP_Gs"),
                 PP_Hs = info.PP_hessians().viewer().name("PP_Hs"),
                 spd_stats,
                 // offsets
                 ee_offset,
                 pe_offset,
                 pp_offset] __device__(IndexT idx) mutable
                {
                    if(idx < ee_offset)  // PT
                    {
                        int      i    = idx;
                        Vector4i PT   = PTs(i);
                        Vector4i cids = {contact_ids(PT[0]),
                                         contact_ids(PT[1]),
                                         contact_ids(PT[2]),
                                         contact_ids(PT[3])};
                        Float    kt2  = PT_kappa(table, cids) * dt * dt;

                        const auto& P  = Ps(PT[0]);
                        const auto& T0 = Ps(PT[1]);
                        const auto& T1 = Ps(PT[2]);
                        const auto& T2 = Ps(PT[3]);

                        Float thickness = PT_thickness(thicknesses(PT(0)),
                                                       thicknesses(PT(1)),
                                                       thicknesses(PT(2)),
                                                       thicknesses(PT(3)));
                        Float d_hat     = PT_d_hat(
                            d_hats(PT(0)), d_hats(PT(1)), d_hats(PT(2)), d_hats(PT(3)));
                        Vector4i flag =
                            distance::point_triangle_distance_flag(P, T0, T1, T2);

                        Vector12 G;
                        if(gradient_only)
                        {
                            PT_barrier_gradient(G, flag, kt2, d_hat, thickness, P, T0, T1, T2);
                            DoubletVectorAssembler DVA{PT_Gs};
                            DVA.segment<4>(i * 4).write(PT, G);
                        }
                        else
                        {
                            Matrix12x12 H;
                            PT_barrier_gradient_hessian(
                                G, H, flag, kt2, d_hat, thickness, P, T0, T1, T2);
                            Matrix12x12 H_before = H;
                            make_spd(H);
                            corex_record_spd_projection(spd_stats, CorexSpdPT, H_before, H);
                            DoubletVectorAssembler DVA{PT_Gs};
                            DVA.segment<4>(i * 4).write(PT, G);
                            TripletMatrixAssembler TMA{PT_Hs};
                            TMA.half_block<4>(i * PTHalfHessianSize).write(PT, H);
                        }
                    }
                    else if(idx < pe_offset)  // EE
                    {
                        int      i    = idx - ee_offset;
                        Vector4i EE   = EEs(i);
                        Vector4i cids = {contact_ids(EE[0]),
                                         contact_ids(EE[1]),
                                         contact_ids(EE[2]),
                                         contact_ids(EE[3])};
                        Float    kt2  = EE_kappa(table, cids) * dt * dt;

                        const auto& E0     = Ps(EE[0]);
                        const auto& E1     = Ps(EE[1]);
                        const auto& E2     = Ps(EE[2]);
                        const auto& E3     = Ps(EE[3]);
                        const auto& t0_Ea0 = rest_Ps(EE[0]);
                        const auto& t0_Ea1 = rest_Ps(EE[1]);
                        const auto& t0_Eb0 = rest_Ps(EE[2]);
                        const auto& t0_Eb1 = rest_Ps(EE[3]);

                        Float thickness = EE_thickness(thicknesses(EE(0)),
                                                       thicknesses(EE(1)),
                                                       thicknesses(EE(2)),
                                                       thicknesses(EE(3)));
                        Float d_hat     = EE_d_hat(
                            d_hats(EE(0)), d_hats(EE(1)), d_hats(EE(2)), d_hats(EE(3)));
                        Vector4i flag = distance::edge_edge_distance_flag(E0, E1, E2, E3);

                        Vector12 G;
                        if(gradient_only)
                        {
                            mollified_EE_barrier_gradient(
                                G, flag, kt2, d_hat, thickness, t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, E0, E1, E2, E3);
                            DoubletVectorAssembler DVA{EE_Gs};
                            DVA.segment<4>(i * 4).write(EE, G);
                        }
                        else
                        {
                            Matrix12x12 H;
                            mollified_EE_barrier_gradient_hessian(
                                G, H, flag, kt2, d_hat, thickness, t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, E0, E1, E2, E3);
                            Matrix12x12 H_before = H;
                            make_spd(H);
                            corex_record_spd_projection(spd_stats, CorexSpdEE, H_before, H);
                            DoubletVectorAssembler DVA{EE_Gs};
                            DVA.segment<4>(i * 4).write(EE, G);
                            TripletMatrixAssembler TMA{EE_Hs};
                            TMA.half_block<4>(i * EEHalfHessianSize).write(EE, H);
                        }
                    }
                    else if(idx < pp_offset)  // PE
                    {
                        int      i    = idx - pe_offset;
                        Vector3i PE   = PEs(i);
                        Vector3i cids = {contact_ids(PE[0]),
                                         contact_ids(PE[1]),
                                         contact_ids(PE[2])};
                        Float    kt2  = PE_kappa(table, cids) * dt * dt;

                        const auto& P  = Ps(PE[0]);
                        const auto& E0 = Ps(PE[1]);
                        const auto& E1 = Ps(PE[2]);

                        Float thickness = PE_thickness(thicknesses(PE(0)),
                                                       thicknesses(PE(1)),
                                                       thicknesses(PE(2)));
                        Float d_hat =
                            PE_d_hat(d_hats(PE(0)), d_hats(PE(1)), d_hats(PE(2)));
                        Vector3i flag = distance::point_edge_distance_flag(P, E0, E1);

                        Vector9 G;
                        if(gradient_only)
                        {
                            PE_barrier_gradient(G, flag, kt2, d_hat, thickness, P, E0, E1);
                            DoubletVectorAssembler DVA{PE_Gs};
                            DVA.segment<3>(i * 3).write(PE, G);
                        }
                        else
                        {
                            Matrix9x9 H;
                            PE_barrier_gradient_hessian(
                                G, H, flag, kt2, d_hat, thickness, P, E0, E1);
                            Matrix9x9 H_before = H;
                            make_spd(H);
                            corex_record_spd_projection(spd_stats, CorexSpdPE, H_before, H);
                            DoubletVectorAssembler DVA{PE_Gs};
                            DVA.segment<3>(i * 3).write(PE, G);
                            TripletMatrixAssembler TMA{PE_Hs};
                            TMA.half_block<3>(i * PEHalfHessianSize).write(PE, H);
                        }
                    }
                    else  // PP
                    {
                        int         i  = idx - pp_offset;
                        const auto& PP = PPs(i);
                        Vector2i cids = {contact_ids(PP[0]), contact_ids(PP[1])};
                        Float kt2 = PP_kappa(table, cids) * dt * dt;

                        const auto& P0 = Ps(PP[0]);
                        const auto& P1 = Ps(PP[1]);

                        Float thickness =
                            PP_thickness(thicknesses(PP(0)), thicknesses(PP(1)));
                        Float d_hat = PP_d_hat(d_hats(PP(0)), d_hats(PP(1)));
                        Vector2i flag = distance::point_point_distance_flag(P0, P1);

                        Vector6 G;
                        if(gradient_only)
                        {
                            PP_barrier_gradient(G, flag, kt2, d_hat, thickness, P0, P1);
                            DoubletVectorAssembler DVA{PP_Gs};
                            DVA.segment<2>(i * 2).write(PP, G);
                        }
                        else
                        {
                            Matrix6x6 H;
                            PP_barrier_gradient_hessian(
                                G, H, flag, kt2, d_hat, thickness, P0, P1);
                            Matrix6x6 H_before = H;
                            make_spd(H);
                            corex_record_spd_projection(spd_stats, CorexSpdPP, H_before, H);
                            DoubletVectorAssembler DVA{PP_Gs};
                            DVA.segment<2>(i * 2).write(PP, G);
                            TripletMatrixAssembler TMA{PP_Hs};
                            TMA.half_block<2>(i * PPHalfHessianSize).write(PP, H);
                        }
                    }
                });

        if(spd_diag_enabled)
        {
            cudaDeviceSynchronize();
            CorexContactSpdStats h_stats{};
            cudaMemcpy(&h_stats, spd_stats, sizeof(CorexContactSpdStats), cudaMemcpyDeviceToHost);
            const char* names[4] = {"PT", "EE", "PE", "PP"};
            for(int t = 0; t < 4; ++t)
            {
                auto count = h_stats.count[t];
                double inv_count = count > 0 ? 1.0 / static_cast<double>(count) : 0.0;
                spdlog::info("[corex_spd_contact] type={} count={} "
                             "projected={} projected_rate={:.3f} "
                             "gershgorin_lower_avg={:.9g} correction_avg={:.9g} "
                             "correction_max={:.9g} diag_abs_avg={:.9g} "
                             "correction_diag_ratio_avg={:.9g} correction_diag_ratio_max={:.9g}",
                             names[t],
                             count,
                             h_stats.projected[t],
                             100.0 * static_cast<double>(h_stats.projected[t]) * inv_count,
                             static_cast<double>(h_stats.lower_sum[t]) * inv_count,
                             static_cast<double>(h_stats.correction_sum[t]) * inv_count,
                             static_cast<double>(h_stats.max_correction[t]),
                             static_cast<double>(h_stats.diag_abs_sum[t]) * inv_count,
                             static_cast<double>(h_stats.correction_diag_ratio_sum[t]) * inv_count,
                             static_cast<double>(h_stats.max_correction_diag_ratio[t]));
            }
            cudaFree(spd_stats);
        }
    }
};

REGISTER_SIM_SYSTEM(IPCSimplexNormalContact);
}  // namespace uipc::backend::cuda
#endif
