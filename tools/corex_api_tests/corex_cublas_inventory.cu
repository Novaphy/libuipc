/**
 * corex_cublas_inventory — A-class symbols from api_test_matrix.md (一符号一测).
 *
 * A checklist: cublasCreate, cublasDestroy, cublasSetStream, cublasSetPointerMode,
 *              cublasDotEx, cublasSdot, cublasSnrm2.
 * Non-A (inventory only): cublasDdot, cublasDnrm2 (B), cublasNrm2Ex (D if <11).
 *
 * Correctness: length-64 deterministic vectors (non-trivial accumulation) + stride-16
 * (incx=2, incy=3 for dot; incx=2 for nrm2) cross-checked vs CPU in double/float.
 */
#include "correctness_utils.hpp"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdio>
#include <cstring>

namespace
{
bool cublas_ok(cublasStatus_t s, const char* what)
{
    if(s == CUBLAS_STATUS_SUCCESS)
        return true;
    if(s == CUBLAS_STATUS_NOT_SUPPORTED)
    {
        std::fprintf(stderr, "[WARN] %s: NOT_SUPPORTED\n", what);
        ++g_warnings();
        return false;
    }
    std::fprintf(stderr, "[FAIL] %s: status=%d\n", what, static_cast<int>(s));
    ++g_failures();
    return false;
}

double cpu_norm2_sq(const double* x, int n)
{
    double s = 0;
    for(int i = 0; i < n; ++i)
        s += x[i] * x[i];
    return s;
}

double cpu_dot_stride_d(const double* x, const double* y, int n, int incx, int incy)
{
    double s = 0;
    for(int i = 0; i < n; ++i)
        s += x[i * incx] * y[i * incy];
    return s;
}

float cpu_norm2_stride_f(const float* x, int n, int incx)
{
    double s = 0;
    for(int i = 0; i < n; ++i)
    {
        double v = static_cast<double>(x[i * incx]);
        s += v * v;
    }
    return static_cast<float>(std::sqrt(s));
}

/** Deterministic non-trivial vectors (reproducible). */
void fill_vectors_main(int n, double* hx, double* hy)
{
    for(int i = 0; i < n; ++i)
    {
        const double t = static_cast<double>(i + 1);
        hx[i] = std::sin(0.17 * t) + 0.01 * static_cast<double>(i);
        hy[i] = std::cos(0.31 * t) - 0.02 * static_cast<double>(i);
    }
}

bool test_cublasCreate(cublasHandle_t* h)
{
    case_begin("cublasCreate");
    return cublas_ok(cublasCreate(h), "cublasCreate");
}

bool test_cublasSetStream(cublasHandle_t h, cudaStream_t stream)
{
    case_begin("cublasSetStream");
    return cublas_ok(cublasSetStream(h, stream), "cublasSetStream");
}

bool test_cublasSetPointerMode(cublasHandle_t h)
{
    case_begin("cublasSetPointerMode");
    if(!cublas_ok(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode HOST"))
        return false;
    return cublas_ok(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode HOST (again)");
}

/** DotEx: unit stride (n=N) + strided (n=Ns, incx=2, incy=3) inside one CASE. */
bool test_cublasDotEx(cublasHandle_t h, int n, float* sdx, float* sdy, double dot_ref, int n_s, float* sdx_s,
                      float* sdy_s, double dot_ref_s)
{
    case_begin("cublasDotEx");
    bool all = true;
    {
        float            dot_ex = 0;
        cublasStatus_t stex =
            cublasDotEx(h, n, sdx, CUDA_R_32F, 1, sdy, CUDA_R_32F, 1, &dot_ex, CUDA_R_32F, CUDA_R_32F);
        if(stex == CUBLAS_STATUS_NOT_SUPPORTED)
        {
            warn("cublasDotEx NOT_SUPPORTED (expected on some stacks; A-class success path N/A here)");
            return true;
        }
        if(!cublas_ok(stex, "cublasDotEx stride1"))
            all = false;
        else
        {
            float ref = static_cast<float>(dot_ref);
            if(!expect_near_f("cublasDotEx stride1 value", dot_ex, ref, 2e-4f))
                all = false;
        }
    }
    {
        float            dot_ex = 0;
        const int        incx = 2, incy = 3;
        cublasStatus_t stex = cublasDotEx(h, n_s, sdx_s, CUDA_R_32F, incx, sdy_s, CUDA_R_32F, incy, &dot_ex,
                                          CUDA_R_32F, CUDA_R_32F);
        if(stex == CUBLAS_STATUS_NOT_SUPPORTED)
        {
            warn("cublasDotEx strided NOT_SUPPORTED");
            return all;
        }
        if(!cublas_ok(stex, "cublasDotEx strided"))
            all = false;
        else if(!expect_near_f("cublasDotEx strided value", dot_ex, static_cast<float>(dot_ref_s), 2e-4f))
            all = false;
    }
    return all;
}

bool test_cublasSdot(cublasHandle_t h, int n, float* sdx, float* sdy, double dot_ref, int n_s, float* sdx_s,
                     float* sdy_s, double dot_ref_s)
{
    case_begin("cublasSdot");
    bool all = true;
    {
        float dot_s = 0;
        if(!cublas_ok(cublasSdot(h, n, sdx, 1, sdy, 1, &dot_s), "cublasSdot stride1"))
            all = false;
        else if(!expect_near_f("cublasSdot stride1 value", dot_s, static_cast<float>(dot_ref), 2e-4f))
            all = false;
    }
    {
        float          dot_s = 0;
        const int      incx = 2, incy = 3;
        if(!cublas_ok(cublasSdot(h, n_s, sdx_s, incx, sdy_s, incy, &dot_s), "cublasSdot strided"))
            all = false;
        else if(!expect_near_f("cublasSdot strided value", dot_s, static_cast<float>(dot_ref_s), 2e-4f))
            all = false;
    }
    return all;
}

bool test_cublasSnrm2(cublasHandle_t h, int n, const float* sfx, float* sdx, int n_s, float* sdx_s, int incx_s,
                      float nrm_ref_s)
{
    case_begin("cublasSnrm2");
    bool all = true;
    {
        float snrm_ref = 0;
        for(int i = 0; i < n; ++i)
            snrm_ref += sfx[i] * sfx[i];
        snrm_ref = std::sqrt(snrm_ref);
        float snrm = 0;
        if(!cublas_ok(cublasSnrm2(h, n, sdx, 1, &snrm), "cublasSnrm2 stride1"))
            all = false;
        else if(!expect_near_f("cublasSnrm2 stride1 value", snrm, snrm_ref, 2e-4f))
            all = false;
    }
    {
        float snrm = 0;
        if(!cublas_ok(cublasSnrm2(h, n_s, sdx_s, incx_s, &snrm), "cublasSnrm2 strided"))
            all = false;
        else if(!expect_near_f("cublasSnrm2 strided value", snrm, nrm_ref_s, 2e-4f))
            all = false;
    }
    return all;
}

void test_cublasDdot_inventory_only(cublasHandle_t h, int n, double* dx, double* dy, double dot_ref, int n_s,
                                    double* dx_s, double* dy_s, double dot_ref_s)
{
    case_begin("cublasDdot");
    {
        double dot_gpu = 0;
        if(cublas_ok(cublasDdot(h, n, dx, 1, dy, 1, &dot_gpu), "cublasDdot stride1"))
            expect_near_d("cublasDdot stride1 (inventory)", dot_gpu, dot_ref, 1e-13);
    }
    {
        double         dot_gpu = 0;
        const int      incx = 2, incy = 3;
        if(cublas_ok(cublasDdot(h, n_s, dx_s, incx, dy_s, incy, &dot_gpu), "cublasDdot strided"))
            expect_near_d("cublasDdot strided (inventory)", dot_gpu, dot_ref_s, 1e-13);
    }
}

void test_cublasDnrm2_inventory_only(cublasHandle_t h, int n, const double* hx, double* dx, int n_s, double* dx_s,
                                     int incx_s, double nrm_ref_s)
{
    case_begin("cublasDnrm2");
    {
        double dnrm = 0;
        if(cublas_ok(cublasDnrm2(h, n, dx, 1, &dnrm), "cublasDnrm2 stride1"))
            expect_near_d("cublasDnrm2 stride1 (inventory)", dnrm, std::sqrt(cpu_norm2_sq(hx, n)), 1e-12);
    }
    {
        double dnrm = 0;
        if(cublas_ok(cublasDnrm2(h, n_s, dx_s, incx_s, &dnrm), "cublasDnrm2 strided"))
            expect_near_d("cublasDnrm2 strided (inventory)", dnrm, nrm_ref_s, 1e-12);
    }
}

bool test_cublasDestroy(cublasHandle_t h)
{
    case_begin("cublasDestroy");
    return cublas_ok(cublasDestroy(h), "cublasDestroy");
}

}  // namespace

int main()
{
    std::printf("=== corex_cublas_inventory ===\n");

    if(!cuda_ok(cudaSetDevice(0), "cudaSetDevice"))
        return 1;

    cudaStream_t stream{};
    if(!cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags"))
        return 1;

    cublasHandle_t h{};
    if(!test_cublasCreate(&h))
    {
        cudaStreamDestroy(stream);
        return 1;
    }

    test_cublasSetStream(h, stream);
    test_cublasSetPointerMode(h);

    constexpr int N  = 64;
    constexpr int Ns = 16;
    constexpr int IX = 2;
    constexpr int IY = 3;

    double hx[N], hy[N];
    fill_vectors_main(N, hx, hy);
    double dot_ref = 0;
    for(int i = 0; i < N; ++i)
        dot_ref += hx[i] * hy[i];

    constexpr int LEN_FX_S = 1 + (Ns - 1) * IX;
    constexpr int LEN_FY_S = 1 + (Ns - 1) * IY;
    float           sfx_s[LEN_FX_S]{};
    float           sfy_s[LEN_FY_S]{};
    double          hdx_s[LEN_FX_S]{};
    double          hdy_s[LEN_FY_S]{};
    for(int i = 0; i < LEN_FX_S; ++i)
        sfx_s[i] = 0.f;
    for(int i = 0; i < LEN_FY_S; ++i)
        sfy_s[i] = 0.f;
    for(int i = 0; i < Ns; ++i)
    {
        const double t = static_cast<double>(i + 1);
        sfx_s[i * IX] = static_cast<float>(std::sin(0.41 * t));
        sfy_s[i * IY] = static_cast<float>(std::cos(0.19 * t));
        hdx_s[i * IX] = static_cast<double>(sfx_s[i * IX]);
        hdy_s[i * IY] = static_cast<double>(sfy_s[i * IY]);
    }
    double       dot_ref_s = cpu_dot_stride_d(hdx_s, hdy_s, Ns, IX, IY);
    const float  nrm_ref_f_s = cpu_norm2_stride_f(sfx_s, Ns, IX);
    const double nrm_ref_d_s =
        std::sqrt(cpu_dot_stride_d(hdx_s, hdx_s, Ns, IX, IX));  // same as norm of x with stride

    float  sfx[N], sfy[N];
    for(int i = 0; i < N; ++i)
    {
        sfx[i] = static_cast<float>(hx[i]);
        sfy[i] = static_cast<float>(hy[i]);
    }

    double* dx = nullptr;
    double* dy = nullptr;
    float*  sdx = nullptr;
    float*  sdy = nullptr;
    double* dx_s = nullptr;
    double* dy_s = nullptr;
    float*  sdx_s = nullptr;
    float*  sdy_s = nullptr;
    cuda_ok(cudaMalloc(&dx, sizeof(double) * N), "cudaMalloc dx");
    cuda_ok(cudaMalloc(&dy, sizeof(double) * N), "cudaMalloc dy");
    cuda_ok(cudaMalloc(&sdx, sizeof(float) * N), "cudaMalloc sdx");
    cuda_ok(cudaMalloc(&sdy, sizeof(float) * N), "cudaMalloc sdy");
    cuda_ok(cudaMalloc(&dx_s, sizeof(double) * LEN_FX_S), "cudaMalloc dx_s");
    cuda_ok(cudaMalloc(&dy_s, sizeof(double) * LEN_FY_S), "cudaMalloc dy_s");
    cuda_ok(cudaMalloc(&sdx_s, sizeof(float) * LEN_FX_S), "cudaMalloc sdx_s");
    cuda_ok(cudaMalloc(&sdy_s, sizeof(float) * LEN_FY_S), "cudaMalloc sdy_s");

    cuda_ok(cudaMemcpy(dx, hx, sizeof(hx), cudaMemcpyHostToDevice), "cudaMemcpy dx");
    cuda_ok(cudaMemcpy(dy, hy, sizeof(hy), cudaMemcpyHostToDevice), "cudaMemcpy dy");
    cuda_ok(cudaMemcpy(sdx, sfx, sizeof(sfx), cudaMemcpyHostToDevice), "cudaMemcpy sdx");
    cuda_ok(cudaMemcpy(sdy, sfy, sizeof(sfy), cudaMemcpyHostToDevice), "cudaMemcpy sdy");
    cuda_ok(cudaMemcpy(dx_s, hdx_s, sizeof(double) * LEN_FX_S, cudaMemcpyHostToDevice), "cudaMemcpy dx_s");
    cuda_ok(cudaMemcpy(dy_s, hdy_s, sizeof(double) * LEN_FY_S, cudaMemcpyHostToDevice), "cudaMemcpy dy_s");
    cuda_ok(cudaMemcpy(sdx_s, sfx_s, sizeof(sfx_s), cudaMemcpyHostToDevice), "cudaMemcpy sdx_s");
    cuda_ok(cudaMemcpy(sdy_s, sfy_s, sizeof(sfy_s), cudaMemcpyHostToDevice), "cudaMemcpy sdy_s");

    test_cublasDotEx(h, N, sdx, sdy, dot_ref, Ns, sdx_s, sdy_s, dot_ref_s);
    test_cublasSdot(h, N, sdx, sdy, dot_ref, Ns, sdx_s, sdy_s, dot_ref_s);
    test_cublasSnrm2(h, N, sfx, sdx, Ns, sdx_s, IX, nrm_ref_f_s);

    test_cublasDdot_inventory_only(h, N, dx, dy, dot_ref, Ns, dx_s, dy_s, dot_ref_s);
    test_cublasDnrm2_inventory_only(h, N, hx, dx, Ns, dx_s, IX, nrm_ref_d_s);

#if CUDART_VERSION >= 11000
    case_begin("cublasNrm2Ex");
    {
        double nrm2_ex_d = 0;
        cublasStatus_t n2ex =
            cublasNrm2Ex(h, N, dx, CUDA_R_64F, 1, &nrm2_ex_d, CUDA_R_64F, CUDA_R_64F);
        if(n2ex == CUBLAS_STATUS_NOT_SUPPORTED)
            warn("cublasNrm2Ex NOT_SUPPORTED");
        else if(cublas_ok(n2ex, "cublasNrm2Ex stride1"))
            expect_near_d("cublasNrm2Ex value stride1", nrm2_ex_d, std::sqrt(cpu_norm2_sq(hx, N)), 1e-12);

        n2ex = cublasNrm2Ex(h, Ns, dx_s, CUDA_R_64F, IX, &nrm2_ex_d, CUDA_R_64F, CUDA_R_64F);
        if(n2ex == CUBLAS_STATUS_NOT_SUPPORTED)
            warn("cublasNrm2Ex strided NOT_SUPPORTED");
        else if(cublas_ok(n2ex, "cublasNrm2Ex strided"))
            expect_near_d("cublasNrm2Ex value strided", nrm2_ex_d, nrm_ref_d_s, 1e-12);

        float nrm2_ex_f = 0;
        float snrm_ref = 0;
        for(int i = 0; i < N; ++i)
            snrm_ref += sfx[i] * sfx[i];
        snrm_ref = std::sqrt(snrm_ref);
        n2ex = cublasNrm2Ex(h, N, sdx, CUDA_R_32F, 1, &nrm2_ex_f, CUDA_R_32F, CUDA_R_32F);
        if(n2ex == CUBLAS_STATUS_NOT_SUPPORTED)
            warn("cublasNrm2Ex float NOT_SUPPORTED");
        else if(cublas_ok(n2ex, "cublasNrm2Ex float stride1"))
            expect_near_f("cublasNrm2Ex float value stride1", nrm2_ex_f, snrm_ref, 2e-4f);

        n2ex = cublasNrm2Ex(h, Ns, sdx_s, CUDA_R_32F, IX, &nrm2_ex_f, CUDA_R_32F, CUDA_R_32F);
        if(n2ex == CUBLAS_STATUS_NOT_SUPPORTED)
            warn("cublasNrm2Ex float strided NOT_SUPPORTED");
        else if(cublas_ok(n2ex, "cublasNrm2Ex float strided"))
            expect_near_f("cublasNrm2Ex float value strided", nrm2_ex_f, nrm_ref_f_s, 2e-4f);
    }
#else
    warn("cublasNrm2Ex skipped (CUDART_VERSION < 11000; symbol may be absent from libcublas)");
#endif

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(sdx);
    cudaFree(sdy);
    cudaFree(dx_s);
    cudaFree(dy_s);
    cudaFree(sdx_s);
    cudaFree(sdy_s);

    test_cublasDestroy(h);
    cudaStreamDestroy(stream);

    print_summary("corex_cublas_inventory");
    return g_failures() > 0 ? 1 : 0;
}
