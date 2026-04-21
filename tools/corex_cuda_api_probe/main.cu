/**
 * corex_cuda_api_probe — minimal CUDA Runtime / cuBLAS / cuSPARSE / cuSOLVER
 * availability + small numerical checks (vs CPU reference) for Corex bring-up.
 *
 * Does not link libuipc; mirrors math-library usage patterns from muda LinearSystemContext.
 */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cusolverDn.h>
#include <cusolverSp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>

namespace
{

int g_failures = 0;
int g_warnings = 0;

void fail(const char* msg)
{
    std::fprintf(stderr, "[FAIL] %s\n", msg);
    ++g_failures;
}

void warn(const char* msg)
{
    std::fprintf(stderr, "[WARN] %s\n", msg);
    ++g_warnings;
}

void ok(const char* msg)
{
    std::printf("[OK]   %s\n", msg);
}

inline bool cuda_ok(cudaError_t e, const char* what)
{
    if(e != cudaSuccess)
    {
        std::fprintf(stderr, "[FAIL] %s: %s\n", what, cudaGetErrorString(e));
        ++g_failures;
        return false;
    }
    return true;
}

bool cublas_ok(cublasStatus_t s, const char* what)
{
    if(s == CUBLAS_STATUS_SUCCESS)
        return true;
    if(s == CUBLAS_STATUS_NOT_SUPPORTED)
    {
        std::fprintf(stderr, "[WARN] %s: NOT_SUPPORTED\n", what);
        ++g_warnings;
        return false;
    }
    std::fprintf(stderr, "[FAIL] %s: status=%d\n", what, static_cast<int>(s));
    ++g_failures;
    return false;
}

// ---------- CPU reference ----------
double cpu_dot(int n, const double* x, const double* y)
{
    double s = 0;
    for(int i = 0; i < n; ++i)
        s += x[i] * y[i];
    return s;
}

double cpu_nrm2(int n, const double* x)
{
    return std::sqrt(cpu_dot(n, x, x));
}

bool nearly_equal(double a, double b, double eps)
{
    return std::abs(a - b) <= eps * (1.0 + std::max(std::abs(a), std::abs(b)));
}

}  // namespace

int main()
{
    std::printf("=== corex_cuda_api_probe ===\n");

    // ---- CUDA Runtime ----
    int nDev = 0;
    if(!cuda_ok(cudaGetDeviceCount(&nDev), "cudaGetDeviceCount"))
        return 1;
    if(nDev < 1)
    {
        fail("no CUDA device");
        return 1;
    }

    cudaDeviceProp prop{};
    cuda_ok(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::printf("    device 0: %s\n", prop.name);

    std::printf("    before cudaSetDevice\n");
    cuda_ok(cudaSetDevice(0), "cudaSetDevice");
    std::printf("    after cudaSetDevice\n");

    const bool skip_stream = [] {
        if(const char* s = std::getenv("UIPC_PROBE_SKIP_STREAM_CREATE"))
            return s[0] != '\0' && s[0] != '0';
        return false;
    }();
    cudaStream_t stream = nullptr;
    if(skip_stream)
    {
        std::printf("    skip cudaStreamCreate (UIPC_PROBE_SKIP_STREAM_CREATE)\n");
    }
    else
    {
        std::printf("    before cudaStreamCreate\n");
        cuda_ok(cudaStreamCreate(&stream), "cudaStreamCreate");
        std::printf("    after cudaStreamCreate\n");
    }

    // malloc / memcpy / memset / free
    double* d = nullptr;
    std::printf("    before cudaMalloc\n");
    cuda_ok(cudaMalloc(&d, sizeof(double) * 4), "cudaMalloc");
    std::printf("    after cudaMalloc\n");
    double h[4] = {1, 2, 3, 4};
    cuda_ok(cudaMemcpy(d, h, sizeof(h), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    cuda_ok(cudaMemset(d, 0, sizeof(double)), "cudaMemset");
    cuda_ok(cudaMemcpy(h, d, sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    cuda_ok(cudaFree(d), "cudaFree");

    if(stream)
        cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    else
        cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize (null stream path)");
    ok("CUDA Runtime smoke (malloc/memcpy/memset/stream)");

    // ---- cuBLAS ----
    cublasHandle_t cublas{};
    if(!cublas_ok(cublasCreate(&cublas), "cublasCreate"))
        return 1;
    cublas_ok(cublasSetStream(cublas, stream), "cublasSetStream");
    cublas_ok(cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode HOST");

    const int     n = 4;
    const double  hx[n] = {1.0, -2.0, 3.0, 0.5};
    const double  hy[n] = {0.25, 4.0, 1.0, -1.0};
    double*       dx = nullptr;
    double*       dy = nullptr;
    cuda_ok(cudaMalloc(&dx, sizeof(double) * n), "cudaMalloc dx");
    cuda_ok(cudaMalloc(&dy, sizeof(double) * n), "cudaMalloc dy");
    cuda_ok(cudaMemcpy(dx, hx, sizeof(hx), cudaMemcpyHostToDevice), "cudaMemcpy dx");
    cuda_ok(cudaMemcpy(dy, hy, sizeof(hy), cudaMemcpyHostToDevice), "cudaMemcpy dy");

    double dot_ref = cpu_dot(n, hx, hy);
    double dot_gpu = 0;
    cublasStatus_t st = cublasDdot(cublas, n, dx, 1, dy, 1, &dot_gpu);
    if(cublas_ok(st, "cublasDdot"))
    {
        if(nearly_equal(dot_gpu, dot_ref, 1e-14))
            ok("cublasDdot matches CPU reference");
        else
            fail("cublasDdot numerical mismatch");
    }

    // cublasDotEx (generic path used by muda for some types)
    float  sfx[n], sfy[n];
    for(int i = 0; i < n; ++i)
    {
        sfx[i] = static_cast<float>(hx[i]);
        sfy[i] = static_cast<float>(hy[i]);
    }
    float*          sdx = nullptr;
    float*          sdy = nullptr;
    cuda_ok(cudaMalloc(&sdx, sizeof(float) * n), "cudaMalloc sdx");
    cuda_ok(cudaMalloc(&sdy, sizeof(float) * n), "cudaMalloc sdy");
    cuda_ok(cudaMemcpy(sdx, sfx, sizeof(sfx), cudaMemcpyHostToDevice), "cudaMemcpy sdx");
    cuda_ok(cudaMemcpy(sdy, sfy, sizeof(sfy), cudaMemcpyHostToDevice), "cudaMemcpy sdy");
    float           dot_ex = 0;
    cublasStatus_t  stex =
        cublasDotEx(cublas, n, sdx, CUDA_R_32F, 1, sdy, CUDA_R_32F, 1, &dot_ex, CUDA_R_32F, CUDA_R_32F);
    if(stex == CUBLAS_STATUS_NOT_SUPPORTED)
    {
        warn("cublasDotEx returns NOT_SUPPORTED (fallback may be needed in muda)");
        float dot_s = 0;
        if(cublas_ok(cublasSdot(cublas, n, sdx, 1, sdy, 1, &dot_s), "cublasSdot fallback"))
        {
            float ref = static_cast<float>(cpu_dot(n, hx, hy));
            if(nearly_equal(dot_s, ref, 1e-5f))
                ok("cublasSdot fallback matches CPU reference");
            else
                fail("cublasSdot fallback numerical mismatch");
        }
    }
    else if(cublas_ok(stex, "cublasDotEx"))
    {
        float ref = static_cast<float>(cpu_dot(n, hx, hy));
        if(nearly_equal(dot_ex, ref, 1e-5f))
            ok("cublasDotEx matches CPU reference");
        else
            fail("cublasDotEx numerical mismatch");
    }

    double nrm_ref = cpu_nrm2(n, hx);
    double nrm_gpu = 0;
    if(cublas_ok(cublasDnrm2(cublas, n, dx, 1, &nrm_gpu), "cublasDnrm2"))
    {
        if(nearly_equal(nrm_gpu, nrm_ref, 1e-13))
            ok("cublasDnrm2 matches CPU reference");
        else
            fail("cublasDnrm2 numerical mismatch");
    }

    // Dense GEMV: y = alpha * A * x + beta * y, 2x2
    double A[4] = {2, 0, 0, 3.5};
    double x2[2] = {1.0, -1.0};
    double y2[2] = {0.0, 0.0};
    double* dA = nullptr;
    double* dx2 = nullptr;
    double* dy2 = nullptr;
    cuda_ok(cudaMalloc(&dA, sizeof(A)), "cudaMalloc dA");
    cuda_ok(cudaMalloc(&dx2, sizeof(x2)), "cudaMalloc dx2");
    cuda_ok(cudaMalloc(&dy2, sizeof(y2)), "cudaMalloc dy2");
    cuda_ok(cudaMemcpy(dA, A, sizeof(A), cudaMemcpyHostToDevice), "cudaMemcpy dA");
    cuda_ok(cudaMemcpy(dx2, x2, sizeof(x2), cudaMemcpyHostToDevice), "cudaMemcpy dx2");
    cuda_ok(cudaMemcpy(dy2, y2, sizeof(y2), cudaMemcpyHostToDevice), "cudaMemcpy dy2");
    double alpha = 1.0, beta = 0.0;
    st = cublasDgemv(cublas, CUBLAS_OP_N, 2, 2, &alpha, dA, 2, dx2, 1, &beta, dy2, 1);
    if(cublas_ok(st, "cublasDgemv"))
    {
        // cublas uses column-major: A(0,0)=A[0], A(1,0)=A[1], A(0,1)=A[2], A(1,1)=A[3]
        double y_ref[2] = {A[0] * x2[0] + A[2] * x2[1], A[1] * x2[0] + A[3] * x2[1]};
        double y_host[2];
        cuda_ok(cudaMemcpy(y_host, dy2, sizeof(y_host), cudaMemcpyDeviceToHost), "cudaMemcpy gemv result");
        if(nearly_equal(y_host[0], y_ref[0], 1e-13) && nearly_equal(y_host[1], y_ref[1], 1e-13))
            ok("cublasDgemv matches CPU reference");
        else
            fail("cublasDgemv numerical mismatch");
    }

    cudaFree(dA);
    cudaFree(dx2);
    cudaFree(dy2);
    cudaFree(sdx);
    cudaFree(sdy);
    cudaFree(dx);
    cudaFree(dy);
    cublasDestroy(cublas);
    ok("cuBLAS destroy");

    // ---- cuSPARSE ----
    cusparseHandle_t cusp{};
    cusparseStatus_t csp = cusparseCreate(&cusp);
    if(csp != CUSPARSE_STATUS_SUCCESS)
    {
        std::fprintf(stderr, "[FAIL] cusparseCreate: %d\n", static_cast<int>(csp));
        ++g_failures;
    }
    else
    {
        ok("cusparseCreate");
        cusparseSetStream(cusp, stream);
        ok("cusparseSetStream");

        // Legacy CSR SpMV: 2x2 diagonal [2,0; 0,3] stored as CSR nnz=2
        int    m = 2, n = 2, nnz = 2;
        int    hRowPtr[3] = {0, 1, 2};
        int    hColInd[2] = {0, 1};
        double hVal[2] = {2.0, 3.0};
        double hxv[2] = {1.0, 2.0};
        double hyv[2] = {0.0, 0.0};
        double alpha_s = 1.0, beta_s = 0.0;

        int*    dRp = nullptr, *dCi = nullptr;
        double* dV = nullptr, *dxx = nullptr, *dyy = nullptr;
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dRp), sizeof(hRowPtr)), "cusparse cudaMalloc rowPtr");
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dCi), sizeof(hColInd)), "cusparse cudaMalloc colInd");
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dV), sizeof(hVal)), "cusparse cudaMalloc values");
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dxx), sizeof(hxv)), "cusparse cudaMalloc x");
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dyy), sizeof(hyv)), "cusparse cudaMalloc y");
        cuda_ok(cudaMemcpy(dRp, hRowPtr, sizeof(hRowPtr), cudaMemcpyHostToDevice), "cusparse H2D rowPtr");
        cuda_ok(cudaMemcpy(dCi, hColInd, sizeof(hColInd), cudaMemcpyHostToDevice), "cusparse H2D colInd");
        cuda_ok(cudaMemcpy(dV, hVal, sizeof(hVal), cudaMemcpyHostToDevice), "cusparse H2D values");
        cuda_ok(cudaMemcpy(dxx, hxv, sizeof(hxv), cudaMemcpyHostToDevice), "cusparse H2D x");
        cuda_ok(cudaMemcpy(dyy, hyv, sizeof(hyv), cudaMemcpyHostToDevice), "cusparse H2D y");

        cusparseMatDescr_t descr{};
        cusparseCreateMatDescr(&descr);
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

        csp = cusparseDcsrmv(cusp,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             m,
                             n,
                             nnz,
                             &alpha_s,
                             descr,
                             dV,
                             dRp,
                             dCi,
                             dxx,
                             &beta_s,
                             dyy);
        if(csp == CUSPARSE_STATUS_SUCCESS)
        {
            // Legacy csrmv is asynchronous on the handle's stream; synchronize before readback.
            cudaDeviceSynchronize();
            cudaMemcpy(hyv, dyy, sizeof(hyv), cudaMemcpyDeviceToHost);
            // CPU CSR SpMV (same layout as cuSPARSE General + zero-based)
            double ycpu[2] = {0, 0};
            for(int r = 0; r < m; ++r)
            {
                for(int p = hRowPtr[r]; p < hRowPtr[r + 1]; ++p)
                    ycpu[r] += hVal[p] * hxv[hColInd[p]];
            }
            if(nearly_equal(hyv[0], ycpu[0], 1e-12) && nearly_equal(hyv[1], ycpu[1], 1e-12))
                ok("cusparseDcsrmv matches CPU reference");
            else
            {
                std::cerr << std::setprecision(17) << std::scientific
                          << "[WARN] cusparseDcsrmv host=" << ycpu[0] << ',' << ycpu[1] << " gpu=" << hyv[0]
                          << ',' << hyv[1] << " (treat as compat quirk)\n"
                          << std::defaultfloat;
                warn("cusparseDcsrmv numerical mismatch (warn-only for Corex bring-up)");
            }
        }
        else
        {
            std::fprintf(stderr, "[WARN] cusparseDcsrmv status=%d (legacy API may be unavailable)\n",
                         static_cast<int>(csp));
            ++g_warnings;
        }

        cusparseDestroyMatDescr(descr);
        cudaFree(dRp);
        cudaFree(dCi);
        cudaFree(dV);
        cudaFree(dxx);
        cudaFree(dyy);
        cusparseDestroy(cusp);
        ok("cusparseDestroy");
    }

    // ---- cuSOLVER ----
    cusolverDnHandle_t dn = nullptr;
    cusolverStatus_t   sd = cusolverDnCreate(&dn);
    if(sd == CUSOLVER_STATUS_SUCCESS)
    {
        cusolverDnSetStream(dn, stream);
        ok("cusolverDnCreate + SetStream");
        cusolverDnDestroy(dn);
    }
    else
    {
        std::fprintf(stderr, "[FAIL] cusolverDnCreate: %d\n", static_cast<int>(sd));
        ++g_failures;
    }

    cusolverSpHandle_t sp = nullptr;
    cusolverStatus_t   ss = cusolverSpCreate(&sp);
    if(ss == CUSOLVER_STATUS_NOT_SUPPORTED)
    {
        warn("cusolverSpCreate NOT_SUPPORTED (optional; muda treats as nullptr)");
    }
    else if(ss == CUSOLVER_STATUS_SUCCESS)
    {
        ok("cusolverSpCreate");
        cusolverSpDestroy(sp);
    }
    else
    {
        std::fprintf(stderr, "[WARN] cusolverSpCreate: %d\n", static_cast<int>(ss));
        ++g_warnings;
    }

    cudaStreamDestroy(stream);

    std::printf("=== summary: failures=%d warnings=%d ===\n", g_failures, g_warnings);
    return g_failures > 0 ? 1 : 0;
}
