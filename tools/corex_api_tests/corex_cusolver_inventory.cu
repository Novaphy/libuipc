/**
 * corex_cusolver_inventory — A-class cusolver* from api_test_matrix.md (一符号一测).
 *
 * A checklist: cusolverDnCreate, cusolverDnSetStream, cusolverDnDestroy, cusolverDnDgetrf.
 * B (inventory / WARN): Dpotrf, Dpotrs, Dgetrs, Sp*, etc.
 */
#include "correctness_utils.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace
{
bool cusolver_ok(cusolverStatus_t s, const char* what)
{
    if(s == CUSOLVER_STATUS_SUCCESS)
        return true;
    if(s == CUSOLVER_STATUS_NOT_SUPPORTED)
    {
        std::fprintf(stderr, "[WARN] %s: NOT_SUPPORTED\n", what);
        ++g_warnings();
        return false;
    }
    std::fprintf(stderr, "[FAIL] %s: status=%d\n", what, static_cast<int>(s));
    ++g_failures();
    return false;
}
}  // namespace

int main()
{
    std::printf("=== corex_cusolver_inventory ===\n");

    if(!cuda_ok(cudaSetDevice(0), "cudaSetDevice"))
        return 1;

    cudaStream_t stream{};
    cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");

    cusolverDnHandle_t dn = nullptr;
    {
        case_begin("cusolverDnCreate");
        if(!cusolver_ok(cusolverDnCreate(&dn), "cusolverDnCreate"))
        {
            cudaStreamDestroy(stream);
            return 1;
        }
    }
    {
        case_begin("cusolverDnSetStream");
        cusolver_ok(cusolverDnSetStream(dn, stream), "cusolverDnSetStream");
    }

    // ---------- B-class Potrf path (inventory; no A correctness requirement) ----------
#if CUDART_VERSION >= 11000
    {
        double            hA[] = {4.0, 1.0, 1.0, 4.0};
        double            hb[] = {1.0, 2.0};
        double *          dA = nullptr, *db = nullptr;
        cudaMalloc(&dA, sizeof(hA));
        cudaMalloc(&db, sizeof(hb));
        cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
        cudaMemcpy(db, hb, sizeof(hb), cudaMemcpyHostToDevice);

        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
        int              info = 0;

        const int64_t n64 = 2;
        size_t        d_lwork = 0, h_lwork = 0;
        cusolverStatus_t st = cusolverDnXpotrf_bufferSize(dn, nullptr, uplo, n64, CUDA_R_64F, dA, n64, CUDA_R_64F,
                                                          &d_lwork, &h_lwork);
        case_begin("cusolverDnXpotrf_bufferSize");
        cusolver_ok(st, "cusolverDnXpotrf_bufferSize");
        if(st == CUSOLVER_STATUS_SUCCESS)
        {
            void* dws = nullptr;
            void* hws = nullptr;
            if(d_lwork)
                cudaMalloc(&dws, d_lwork);
            if(h_lwork)
                hws = malloc(h_lwork);
            case_begin("cusolverDnXpotrf");
            st = cusolverDnXpotrf(dn, nullptr, uplo, n64, CUDA_R_64F, dA, n64, CUDA_R_64F, dws, d_lwork, hws, h_lwork,
                                  &info);
            cusolver_ok(st, "cusolverDnXpotrf");
            case_begin("cusolverDnXpotrs");
            if(st == CUSOLVER_STATUS_SUCCESS)
                cusolver_ok(cusolverDnXpotrs(dn, nullptr, uplo, n64, 1, CUDA_R_64F, dA, n64, CUDA_R_64F, db, n64, &info),
                            "cusolverDnXpotrs");
            cudaFree(dws);
            free(hws);
        }
        cudaFree(dA);
        cudaFree(db);
    }
#else
    {
        double            hA[] = {4.0, 1.0, 1.0, 4.0};
        double            hb[] = {1.0, 2.0};
        double *          dA = nullptr, *db = nullptr;
        cudaMalloc(&dA, sizeof(hA));
        cudaMalloc(&db, sizeof(hb));
        cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
        cudaMemcpy(db, hb, sizeof(hb), cudaMemcpyHostToDevice);

        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
        int              info = 0;
        const int        n = 2;
        int              Lwork = 0;
        case_begin("cusolverDnDpotrf_bufferSize");
        cusolverDnDpotrf_bufferSize(dn, uplo, n, dA, n, &Lwork);
        double* work = nullptr;
        cudaMalloc(&work, sizeof(double) * Lwork);
        case_begin("cusolverDnDpotrf");
        cusolver_ok(cusolverDnDpotrf(dn, uplo, n, dA, n, work, Lwork, &info), "cusolverDnDpotrf");
        case_begin("cusolverDnDpotrs");
        cusolver_ok(cusolverDnDpotrs(dn, uplo, n, 1, dA, n, db, n, &info), "cusolverDnDpotrs");
        cudaFree(work);
        cudaFree(dA);
        cudaFree(db);
    }
#endif

    // ---------- A: cusolverDnDgetrf + Dgetrs + CPU reference (8×8, well-conditioned) ----------
    {
        const int   n = 8;
        const size_t n2 = static_cast<size_t>(n) * static_cast<size_t>(n);
        double        hA[64];
        double        hb[8];
        double        x_true[8];
        for(int i = 0; i < n; ++i)
            x_true[i] = 1.0 / static_cast<double>(i + 1);
        for(int i = 0; i < n; ++i)
            for(int j = 0; j < n; ++j)
                hA[i * n + j] =
                    (i == j) ? (12.0 + static_cast<double>(i)) : 0.02 * std::sin(0.3 * static_cast<double>(i * n + j));
        cpu_matvec_rowmajor_d(n, hA, x_true, hb);
        double* dA = nullptr, *db = nullptr;
        cudaMalloc(&dA, n2 * sizeof(double));
        cudaMalloc(&db, static_cast<size_t>(n) * sizeof(double));
        cudaMemcpy(dA, hA, n2 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(db, hb, static_cast<size_t>(n) * sizeof(double), cudaMemcpyHostToDevice);
        int Lwork = 0;
        case_begin("cusolverDnDgetrf_bufferSize");
        cusolverDnDgetrf_bufferSize(dn, n, n, dA, n, &Lwork);
        double* work = nullptr;
        int*    ipiv = nullptr;
        int     info = 0;
        cudaMalloc(&work, sizeof(double) * static_cast<size_t>(Lwork));
        cudaMalloc(&ipiv, sizeof(int) * static_cast<size_t>(n));
        case_begin("cusolverDnDgetrf");
        if(cusolver_ok(cusolverDnDgetrf(dn, n, n, dA, n, work, ipiv, &info), "cusolverDnDgetrf"))
        {
            case_begin("cusolverDnDgetrs");
            if(cusolver_ok(cusolverDnDgetrs(dn, CUBLAS_OP_N, n, 1, dA, n, ipiv, db, n, &info), "cusolverDnDgetrs"))
            {
                cudaDeviceSynchronize();
                double xgpu[8];
                cudaMemcpy(xgpu, db, sizeof(xgpu), cudaMemcpyDeviceToHost);
                double xref[8];
                if(cpu_dense_solve_rowmajor(n, hA, hb, xref))
                {
                    for(int i = 0; i < n; ++i)
                    {
                        char tag[64];
                        std::snprintf(tag, sizeof(tag), "cusolverDnDgetrs x[%d]", i);
                        expect_near_d(tag, xgpu[i], xref[i], 1e-9);
                    }
                }
                else
                    fail("cusolverDnDgetrs CPU ref failed");
            }
        }
        cudaFree(work);
        cudaFree(ipiv);
        cudaFree(dA);
        cudaFree(db);
    }

#if CUDART_VERSION >= 11000
    // Optional X API path (matrix D on 10.x; exercised on 11+).
    {
        cusolverDnParams_t params{};
        case_begin("cusolverDnCreateParams");
        if(cusolver_ok(cusolverDnCreateParams(&params), "cusolverDnCreateParams"))
        {
            case_begin("cusolverDnSetAdvOptions");
            cusolver_ok(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0), "cusolverDnSetAdvOptions");

            const int64_t n = 8;
            double        hA[64];
            double        x_true[8];
            double        hb[8];
            for(int i = 0; i < n; ++i)
                x_true[i] = 1.0 / static_cast<double>(i + 1);
            for(int i = 0; i < n; ++i)
                for(int j = 0; j < n; ++j)
                    hA[i * static_cast<int>(n) + j] =
                        (i == j) ? (12.0 + static_cast<double>(i))
                                 : 0.02 * std::sin(0.3 * static_cast<double>(i * static_cast<int>(n) + j));
            cpu_matvec_rowmajor_d(static_cast<int>(n), hA, x_true, hb);
            double* dA = nullptr, *db = nullptr;
            cudaMalloc(&dA, static_cast<size_t>(n * n) * sizeof(double));
            cudaMalloc(&db, static_cast<size_t>(n) * sizeof(double));
            cudaMemcpy(dA, hA, static_cast<size_t>(n * n) * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(db, hb, static_cast<size_t>(n) * sizeof(double), cudaMemcpyHostToDevice);

            size_t           d_lwork = 0, h_lwork = 0;
            int              info = 0;
            case_begin("cusolverDnXgetrf_bufferSize");
            cusolverDnXgetrf_bufferSize(dn, params, n, n, CUDA_R_64F, dA, n, CUDA_R_64F, &d_lwork, &h_lwork);

            int64_t* dpiv = nullptr;
            cudaMalloc(&dpiv, static_cast<size_t>(n) * sizeof(int64_t));
            void* dws = nullptr;
            if(d_lwork)
                cudaMalloc(&dws, d_lwork);
            void* hws = h_lwork ? malloc(h_lwork) : nullptr;

            case_begin("cusolverDnXgetrf");
            cusolverStatus_t st = cusolverDnXgetrf(dn, params, n, n, CUDA_R_64F, dA, n, dpiv, CUDA_R_64F, dws,
                                                   d_lwork, hws, h_lwork, &info);
            if(cusolver_ok(st, "cusolverDnXgetrf"))
            {
                case_begin("cusolverDnXgetrs");
                st = cusolverDnXgetrs(dn, params, CUBLAS_OP_N, n, 1, CUDA_R_64F, dA, n, dpiv, CUDA_R_64F, db, n, &info);
                if(cusolver_ok(st, "cusolverDnXgetrs"))
                {
                    cudaDeviceSynchronize();
                    double xgpu[8];
                    cudaMemcpy(xgpu, db, sizeof(xgpu), cudaMemcpyDeviceToHost);
                    double xref[8];
                    if(cpu_dense_solve_rowmajor(static_cast<int>(n), hA, hb, xref))
                    {
                        for(int i = 0; i < static_cast<int>(n); ++i)
                        {
                            char tag[64];
                            std::snprintf(tag, sizeof(tag), "cusolverDnXgetrs x[%d]", i);
                            expect_near_d(tag, xgpu[i], xref[i], 1e-9);
                        }
                    }
                    else
                        fail("cusolverDnXgetrs CPU ref failed");
                }
            }

            cudaFree(dpiv);
            cudaFree(dws);
            free(hws);
            cudaFree(dA);
            cudaFree(db);
            case_begin("cusolverDnDestroyParams");
            cusolver_ok(cusolverDnDestroyParams(params), "cusolverDnDestroyParams");
        }
    }
#else
    warn("cusolverDnXgetrf / CreateParams / DestroyParams skipped (CUDART_VERSION < 11000)");
#endif

    {
        case_begin("cusolverDnDestroy");
        cusolver_ok(cusolverDnDestroy(dn), "cusolverDnDestroy");
    }

    // ---------- cusolverSp (B-class create on Corex) ----------
    cusolverSpHandle_t sp = nullptr;
    cusolverStatus_t   ss = cusolverSpCreate(&sp);
    case_begin("cusolverSpCreate");
    if(ss == CUSOLVER_STATUS_NOT_SUPPORTED)
        warn("cusolverSpCreate NOT_SUPPORTED");
    else if(cusolver_ok(ss, "cusolverSpCreate"))
    {
        case_begin("cusolverSpSetStream");
        cusolver_ok(cusolverSpSetStream(sp, stream), "cusolverSpSetStream");

        const int m = 2, nnz = 2;
        int       hRp[] = {0, 1, 2};
        int       hCi[] = {0, 1};
        float     hVa[] = {2.f, 3.f};
        float     hb[] = {2.f, 6.f};
        float     hx[] = {0.f, 0.f};
        int*      dRp = nullptr, *dCi = nullptr;
        float *   dVa = nullptr, *db = nullptr, *dx = nullptr;
        cudaMalloc(&dRp, sizeof(hRp));
        cudaMalloc(&dCi, sizeof(hCi));
        cudaMalloc(&dVa, sizeof(hVa));
        cudaMalloc(&db, sizeof(hb));
        cudaMalloc(&dx, sizeof(hx));
        cudaMemcpy(dRp, hRp, sizeof(hRp), cudaMemcpyHostToDevice);
        cudaMemcpy(dCi, hCi, sizeof(hCi), cudaMemcpyHostToDevice);
        cudaMemcpy(dVa, hVa, sizeof(hVa), cudaMemcpyHostToDevice);
        cudaMemcpy(db, hb, sizeof(hb), cudaMemcpyHostToDevice);

        cusparseMatDescr_t descr{};
        cusparseCreateMatDescr(&descr);
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

        int singularity = 0;
        case_begin("cusolverSpScsrlsvqr");
        ss = cusolverSpScsrlsvqr(sp, m, nnz, descr, dVa, dRp, dCi, db, 1e-7f, 0, dx, &singularity);
        cusolver_ok(ss, "cusolverSpScsrlsvqr");

        double hVd[] = {2.0, 3.0}, hbd[] = {2.0, 6.0}, hxd[] = {0.0, 0.0};
        double *dVd = nullptr, *dbd = nullptr, *dxd = nullptr;
        cudaMalloc(&dVd, sizeof(hVd));
        cudaMalloc(&dbd, sizeof(hbd));
        cudaMalloc(&dxd, sizeof(hxd));
        cudaMemcpy(dVd, hVd, sizeof(hVd), cudaMemcpyHostToDevice);
        cudaMemcpy(dbd, hbd, sizeof(hbd), cudaMemcpyHostToDevice);
        case_begin("cusolverSpDcsrlsvqr");
        ss = cusolverSpDcsrlsvqr(sp, m, nnz, descr, dVd, dRp, dCi, dbd, 1e-12, 0, dxd, &singularity);
        cusolver_ok(ss, "cusolverSpDcsrlsvqr");

        cusparseDestroyMatDescr(descr);
        cudaFree(dRp);
        cudaFree(dCi);
        cudaFree(dVa);
        cudaFree(db);
        cudaFree(dx);
        cudaFree(dVd);
        cudaFree(dbd);
        cudaFree(dxd);

        case_begin("cusolverSpDestroy");
        cusolver_ok(cusolverSpDestroy(sp), "cusolverSpDestroy");
    }

    cudaStreamDestroy(stream);

    print_summary("corex_cusolver_inventory");
    return g_failures() > 0 ? 1 : 0;
}
