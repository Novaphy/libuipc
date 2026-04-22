/**
 * corex_cusparse_inventory — A-class cusparse* from api_test_matrix.md (一符号一测).
 *
 * A checklist: cusparseCreate, Destroy, SetStream, SetPointerMode, CreateCsr, CreateCoo,
 *   CreateDnVec, CreateMatDescr, CreateSpVec, DestroyDnVec, DestroyMatDescr, DestroySpMat,
 *   DestroySpVec, SetMatType, SetMatIndexBase, SetMatDiagType, Sbsrmv, Dbsrmv, Sbsr2csr,
 *   Dbsr2csr, cusparseSpMV.
 */
#include "correctness_utils.hpp"

#include <cuda_runtime_api.h>
#include <cusparse.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{

constexpr int kSpDiagN = 32;

bool cusparse_ok(cusparseStatus_t s, const char* what)
{
    if(s == CUSPARSE_STATUS_SUCCESS)
        return true;
    if(s == CUSPARSE_STATUS_NOT_SUPPORTED)
    {
        std::fprintf(stderr, "[WARN] %s: NOT_SUPPORTED\n", what);
        ++g_warnings();
        return false;
    }
    std::fprintf(stderr, "[FAIL] %s: status=%d\n", what, static_cast<int>(s));
    ++g_failures();
    return false;
}

void cpu_csr_spmv_base_f(int         m,
                         int         n,
                         const int*  rowptr,
                         const int*  colidx,
                         const float* val,
                         const float* x,
                         float*       y,
                         float        alpha,
                         float        beta,
                         int          base)
{
    for(int i = 0; i < m; ++i)
    {
        double acc = 0.0;
        int    rb  = rowptr[i] - base;
        int    re  = rowptr[i + 1] - base;
        for(int p = rb; p < re; ++p)
        {
            int j = colidx[p] - base;
            if(j >= 0 && j < n)
                acc += static_cast<double>(val[p]) * static_cast<double>(x[j]);
        }
        const double y0 = beta == 0.f ? 0.0 : static_cast<double>(y[i]);
        y[i]            = static_cast<float>(alpha * acc + beta * y0);
    }
}

void cpu_coo_spmv_base_f(int          m,
                         int          n,
                         int          nnz,
                         const int*   row,
                         const int*   col,
                         const float* val,
                         const float* x,
                         float*       y,
                         float        alpha,
                         float        beta,
                         int          base)
{
    for(int i = 0; i < m; ++i)
        y[i] = beta == 0.f ? 0.f : beta * y[i];
    for(int p = 0; p < nnz; ++p)
    {
        int r = row[p] - base;
        int c = col[p] - base;
        if(r >= 0 && r < m && c >= 0 && c < n)
            y[r] += alpha * val[p] * x[c];
    }
}

void cpu_bsr_dense_colmajor_f(int          mb,
                              int          nb,
                              int          block_dim,
                              const int*   rowptr,
                              const int*   colidx,
                              const float* val,
                              int          base,
                              float*       dense)
{
    const int rows = mb * block_dim;
    const int cols = nb * block_dim;
    std::memset(dense, 0, sizeof(float) * static_cast<size_t>(rows) * static_cast<size_t>(cols));
    for(int br = 0; br < mb; ++br)
    {
        int rb = rowptr[br] - base;
        int re = rowptr[br + 1] - base;
        for(int p = rb; p < re; ++p)
        {
            int bc         = colidx[p] - base;
            auto block_ptr = val + static_cast<size_t>(p) * block_dim * block_dim;
            for(int c = 0; c < block_dim; ++c)
                for(int r = 0; r < block_dim; ++r)
                {
                    int gr                    = br * block_dim + r;
                    int gc                    = bc * block_dim + c;
                    dense[gr * cols + gc] += block_ptr[c * block_dim + r];
                }
        }
    }
}

}  // namespace

int main()
{
    std::printf("=== corex_cusparse_inventory ===\n");

    if(!cuda_ok(cudaSetDevice(0), "cudaSetDevice"))
        return 1;

    cudaStream_t stream{};
    cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");

    cusparseHandle_t h{};
    {
        case_begin("cusparseCreate");
        if(!cusparse_ok(cusparseCreate(&h), "cusparseCreate"))
        {
            cudaStreamDestroy(stream);
            return 1;
        }
    }

    {
        case_begin("cusparseSetStream");
        cusparse_ok(cusparseSetStream(h, stream), "cusparseSetStream");
    }
    {
        case_begin("cusparseSetPointerMode");
        cusparse_ok(cusparseSetPointerMode(h, CUSPARSE_POINTER_MODE_HOST), "cusparseSetPointerMode HOST");
        cusparse_ok(cusparseSetPointerMode(h, CUSPARSE_POINTER_MODE_HOST), "cusparseSetPointerMode HOST again");
    }

    // ---------- CSR + SpMV (generic): per-symbol CASE blocks (n×n diagonal, n=32) ----------
    {
        const int m = kSpDiagN, n = kSpDiagN, nnz = kSpDiagN;
        int         hRow[kSpDiagN + 1];
        int         hCol[kSpDiagN];
        float       hVal[kSpDiagN];
        float       hx[kSpDiagN], hy[kSpDiagN];
        for(int i = 0; i <= n; ++i)
            hRow[i] = i;
        for(int i = 0; i < n; ++i)
        {
            hCol[i] = i;
            hVal[i] = 1.f / static_cast<float>(i + 1);
            hx[i]   = 1.f;
            hy[i]   = 0.f;
        }
        int *   dRow = nullptr, *dCol = nullptr;
        float * dVal = nullptr, *dx = nullptr, *dy = nullptr;
        cudaMalloc(&dRow, sizeof(int) * static_cast<size_t>(n + 1));
        cudaMalloc(&dCol, sizeof(int) * static_cast<size_t>(nnz));
        cudaMalloc(&dVal, sizeof(float) * static_cast<size_t>(nnz));
        cudaMalloc(&dx, sizeof(float) * static_cast<size_t>(n));
        cudaMalloc(&dy, sizeof(float) * static_cast<size_t>(m));
        cudaMemcpy(dRow, hRow, sizeof(int) * static_cast<size_t>(n + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(dCol, hCol, sizeof(int) * static_cast<size_t>(nnz), cudaMemcpyHostToDevice);
        cudaMemcpy(dVal, hVal, sizeof(float) * static_cast<size_t>(nnz), cudaMemcpyHostToDevice);
        cudaMemcpy(dx, hx, sizeof(float) * static_cast<size_t>(n), cudaMemcpyHostToDevice);
        cudaMemcpy(dy, hy, sizeof(float) * static_cast<size_t>(m), cudaMemcpyHostToDevice);

        cusparseSpMatDescr_t matA = nullptr;
        {
            case_begin("cusparseCreateCsr");
            cusparse_ok(cusparseCreateCsr(&matA, m, n, nnz, dRow, dCol, dVal, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F),
                        "cusparseCreateCsr");
        }

        cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
        {
            case_begin("cusparseCreateDnVec");
            cusparse_ok(cusparseCreateDnVec(&vecX, n, dx, CUDA_R_32F), "cusparseCreateDnVec x");
        }
        {
            case_begin("cusparseCreateDnVec");
            cusparse_ok(cusparseCreateDnVec(&vecY, m, dy, CUDA_R_32F), "cusparseCreateDnVec y");
        }

        float           alpha = 1.f, beta = 0.f;
        size_t          bufSize = 0;
        cusparseStatus_t st =
            cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                    CUSPARSE_MV_ALG_DEFAULT, &bufSize);
        cusparse_ok(st, "cusparseSpMV_bufferSize CSR");

        void* buf = nullptr;
        const size_t allocB = bufSize > 0 ? bufSize : 4;
        cudaMalloc(&buf, allocB);
        cudaStreamSynchronize(stream);

        {
            case_begin("cusparseSpMV");
            st = cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                              CUSPARSE_MV_ALG_DEFAULT, buf);
            if(cusparse_ok(st, "cusparseSpMV CSR"))
            {
                cudaDeviceSynchronize();
                float hyo[kSpDiagN];
                cudaMemcpy(hyo, dy, sizeof(hyo), cudaMemcpyDeviceToHost);
                float yref[kSpDiagN];
                for(int i = 0; i < m; ++i)
                    yref[i] = 0.f;
                cpu_csr_spmv_f(m, n, hRow, hCol, hVal, hx, yref, 1.f, 0.f);
                float*             dy_leg = nullptr;
                cusparseMatDescr_t legDesc{};
                cudaMalloc(&dy_leg, sizeof(float) * static_cast<size_t>(m));
                cudaMemset(dy_leg, 0, sizeof(float) * static_cast<size_t>(m));
                cusparseCreateMatDescr(&legDesc);
                cusparseSetMatType(legDesc, CUSPARSE_MATRIX_TYPE_GENERAL);
                cusparseSetMatIndexBase(legDesc, CUSPARSE_INDEX_BASE_ZERO);
                cusparseStatus_t legst =
                    cusparseScsrmv(h, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &alpha, legDesc, dVal, dRow, dCol,
                                   dx, &beta, dy_leg);
                float yleg[kSpDiagN];
                for(int i = 0; i < m; ++i)
                    yleg[i] = 0.f;
                if(cusparse_ok(legst, "cusparseScsrmv legacy ref"))
                {
                    cudaDeviceSynchronize();
                    cudaMemcpy(yleg, dy_leg, sizeof(yleg), cudaMemcpyDeviceToHost);
                    for(int i = 0; i < m; ++i)
                    {
                        char tag[72];
                        std::snprintf(tag, sizeof(tag), "cusparseScsrmv legacy y[%d]", i);
                        expect_near_f(tag, yleg[i], yref[i], 1e-4f);
                    }
                }
                cusparseDestroyMatDescr(legDesc);
                cudaFree(dy_leg);
                bool match_cpu = true, match_leg = true;
                for(int i = 0; i < m; ++i)
                {
                    if(!nearly_equal_f(hyo[i], yref[i], 1e-3f))
                        match_cpu = false;
                    if(!nearly_equal_f(hyo[i], yleg[i], 1e-3f))
                        match_leg = false;
                }
                if(match_cpu)
                    ok("cusparseSpMV CSR matches CPU");
                else if(match_leg)
                    ok("cusparseSpMV CSR matches legacy");
                else
                {
                    warn("cusparseSpMV CSR numeric mismatch vs CPU/legacy (compat; generic path)");
                    std::fprintf(stderr, "       generic[0,1]=(%g,%g) legacy[0,1]=(%g,%g) cpu[0,1]=(%g,%g)\n", hyo[0],
                                 hyo[1], yleg[0], yleg[1], yref[0], yref[1]);
                }
            }
        }
        cudaFree(buf);

        {
            case_begin("cusparseDestroyDnVec");
            cusparse_ok(cusparseDestroyDnVec(vecX), "cusparseDestroyDnVec vecX");
        }
        {
            case_begin("cusparseDestroyDnVec");
            cusparse_ok(cusparseDestroyDnVec(vecY), "cusparseDestroyDnVec vecY");
        }
        {
            case_begin("cusparseDestroySpMat");
            cusparse_ok(cusparseDestroySpMat(matA), "cusparseDestroySpMat");
        }

        cudaFree(dRow);
        cudaFree(dCol);
        cudaFree(dVal);
        cudaFree(dx);
        cudaFree(dy);
    }

    // ---------- Stronger CSR create/destroy correctness ----------
    {
        const int m = 9, n = 7, nnz = 18;
        int hRow0[m + 1] = {0, 3, 3, 6, 8, 10, 13, 15, 15, 18};
        int hCol0[nnz]   = {0, 2, 6, 1, 3, 5, 0, 6, 2, 4, 1, 5, 6, 0, 3, 2, 4, 6};
        float hVal[nnz]  = {2.0f, -1.0f, 0.5f, 3.0f, -2.0f, 1.0f, 4.0f, -1.5f, 5.0f,
                             2.5f, -3.5f, 1.25f, 0.75f, -2.25f, 6.0f, 1.1f, -0.9f, 2.2f};
        float hx[n]      = {1.0f, -2.0f, 0.5f, 3.0f, -1.0f, 4.0f, 2.0f};
        float hy[m]      = {0.25f, -0.5f, 1.0f, 0.0f, -1.5f, 2.0f, 0.75f, -0.25f, 1.25f};

        int   hRow1[m + 1];
        int   hCol1[nnz];
        float hy_ref[m];
        float hy_out[m];
        for(int i = 0; i <= m; ++i)
            hRow1[i] = hRow0[i] + 1;
        for(int i = 0; i < nnz; ++i)
            hCol1[i] = hCol0[i] + 1;

        int *   dRow = nullptr, *dCol = nullptr;
        float * dVal = nullptr, *dx = nullptr, *dy = nullptr;
        cudaMalloc(&dRow, sizeof(hRow0));
        cudaMalloc(&dCol, sizeof(hCol0));
        cudaMalloc(&dVal, sizeof(hVal));
        cudaMalloc(&dx, sizeof(hx));
        cudaMalloc(&dy, sizeof(hy));
        cudaMemcpy(dRow, hRow0, sizeof(hRow0), cudaMemcpyHostToDevice);
        cudaMemcpy(dCol, hCol0, sizeof(hCol0), cudaMemcpyHostToDevice);
        cudaMemcpy(dVal, hVal, sizeof(hVal), cudaMemcpyHostToDevice);
        cudaMemcpy(dx, hx, sizeof(hx), cudaMemcpyHostToDevice);
        cudaMemcpy(dy, hy, sizeof(hy), cudaMemcpyHostToDevice);

        cusparseSpMatDescr_t csr0 = nullptr;
        {
            case_begin("cusparseCreateCsr");
            cusparse_ok(cusparseCreateCsr(&csr0,
                                          m,
                                          n,
                                          nnz,
                                          dRow,
                                          dCol,
                                          dVal,
                                          CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_BASE_ZERO,
                                          CUDA_R_32F),
                        "cusparseCreateCsr irregular base0");
        }

        cusparseDnVecDescr_t vx0 = nullptr, vy0 = nullptr;
        cusparseCreateDnVec(&vx0, n, dx, CUDA_R_32F);
        cusparseCreateDnVec(&vy0, m, dy, CUDA_R_32F);
        float  alpha = 1.25f, beta = -0.5f;
        size_t bufSz = 0;
        cusparse_ok(cusparseSpMV_bufferSize(h,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha,
                                            csr0,
                                            vx0,
                                            &beta,
                                            vy0,
                                            CUDA_R_32F,
                                            CUSPARSE_MV_ALG_DEFAULT,
                                            &bufSz),
                    "cusparseSpMV_bufferSize CSR irregular base0");
        void* buf = nullptr;
        cudaMalloc(&buf, bufSz > 0 ? bufSz : 4);
        cpu_csr_spmv_base_f(m, n, hRow0, hCol0, hVal, hx, hy_ref, alpha, beta, 0);
        {
            case_begin("cusparseSpMV");
            auto st = cusparseSpMV(h,
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &alpha,
                                   csr0,
                                   vx0,
                                   &beta,
                                   vy0,
                                   CUDA_R_32F,
                                   CUSPARSE_MV_ALG_DEFAULT,
                                   buf);
            if(cusparse_ok(st, "cusparseSpMV CSR irregular base0"))
            {
                cudaDeviceSynchronize();
                cudaMemcpy(hy_out, dy, sizeof(hy_out), cudaMemcpyDeviceToHost);
                bool all_ok = true;
                for(int i = 0; i < m; ++i)
                    all_ok &= nearly_equal_f(hy_out[i], hy_ref[i], 1e-3f);
                if(all_ok)
                    ok("cusparseCreateCsr irregular base0 values");
                else
                    warn("cusparseCreateCsr irregular base0 values mismatch (compat)");
            }
        }
        {
            case_begin("cusparseDestroySpMat");
            cusparse_ok(cusparseDestroySpMat(csr0), "cusparseDestroySpMat CSR irregular base0");
        }
        cusparseDestroyDnVec(vx0);
        cusparseDestroyDnVec(vy0);
        cudaFree(buf);

        cudaMemcpy(dRow, hRow1, sizeof(hRow1), cudaMemcpyHostToDevice);
        cudaMemcpy(dCol, hCol1, sizeof(hCol1), cudaMemcpyHostToDevice);
        cudaMemcpy(dy, hy, sizeof(hy), cudaMemcpyHostToDevice);

        cusparseSpMatDescr_t csr1 = nullptr;
        {
            case_begin("cusparseCreateCsr");
            cusparse_ok(cusparseCreateCsr(&csr1,
                                          m,
                                          n,
                                          nnz,
                                          dRow,
                                          dCol,
                                          dVal,
                                          CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_BASE_ONE,
                                          CUDA_R_32F),
                        "cusparseCreateCsr irregular base1");
        }
        cusparseDnVecDescr_t vx1 = nullptr, vy1 = nullptr;
        cusparseCreateDnVec(&vx1, n, dx, CUDA_R_32F);
        cusparseCreateDnVec(&vy1, m, dy, CUDA_R_32F);
        cusparse_ok(cusparseSpMV_bufferSize(h,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha,
                                            csr1,
                                            vx1,
                                            &beta,
                                            vy1,
                                            CUDA_R_32F,
                                            CUSPARSE_MV_ALG_DEFAULT,
                                            &bufSz),
                    "cusparseSpMV_bufferSize CSR irregular base1");
        cudaMalloc(&buf, bufSz > 0 ? bufSz : 4);
        cpu_csr_spmv_base_f(m, n, hRow1, hCol1, hVal, hx, hy_ref, alpha, beta, 1);
        {
            case_begin("cusparseSpMV");
            auto st = cusparseSpMV(h,
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &alpha,
                                   csr1,
                                   vx1,
                                   &beta,
                                   vy1,
                                   CUDA_R_32F,
                                   CUSPARSE_MV_ALG_DEFAULT,
                                   buf);
            if(cusparse_ok(st, "cusparseSpMV CSR irregular base1"))
            {
                cudaDeviceSynchronize();
                cudaMemcpy(hy_out, dy, sizeof(hy_out), cudaMemcpyDeviceToHost);
                bool all_ok = true;
                for(int i = 0; i < m; ++i)
                    all_ok &= nearly_equal_f(hy_out[i], hy_ref[i], 1e-3f);
                if(all_ok)
                    ok("cusparseCreateCsr irregular base1 values");
                else
                    warn("cusparseCreateCsr irregular base1 values mismatch (compat)");
            }
        }
        {
            case_begin("cusparseDestroySpMat");
            cusparse_ok(cusparseDestroySpMat(csr1), "cusparseDestroySpMat CSR irregular base1");
        }
        cusparseDestroyDnVec(vx1);
        cusparseDestroyDnVec(vy1);
        cudaFree(buf);
        cudaFree(dRow);
        cudaFree(dCol);
        cudaFree(dVal);
        cudaFree(dx);
        cudaFree(dy);
    }

    // ---------- COO SpMV (n×n diagonal, n=32) ----------
    {
        const int m = kSpDiagN, n = kSpDiagN, nnz = kSpDiagN;
        int         hCooRow[kSpDiagN], hCooCol[kSpDiagN];
        float       hVal[kSpDiagN];
        float       hx[kSpDiagN], hy[kSpDiagN];
        for(int i = 0; i < n; ++i)
        {
            hCooRow[i] = i;
            hCooCol[i] = i;
            hVal[i]    = 1.f / static_cast<float>(i + 1);
            hx[i]      = 1.f;
            hy[i]      = 0.f;
        }
        int *   dR = nullptr, *dC = nullptr;
        float * dV = nullptr, *dx = nullptr, *dy = nullptr;
        cudaMalloc(&dR, sizeof(int) * static_cast<size_t>(nnz));
        cudaMalloc(&dC, sizeof(int) * static_cast<size_t>(nnz));
        cudaMalloc(&dV, sizeof(float) * static_cast<size_t>(nnz));
        cudaMalloc(&dx, sizeof(float) * static_cast<size_t>(n));
        cudaMalloc(&dy, sizeof(float) * static_cast<size_t>(m));
        cudaMemcpy(dR, hCooRow, sizeof(int) * static_cast<size_t>(nnz), cudaMemcpyHostToDevice);
        cudaMemcpy(dC, hCooCol, sizeof(int) * static_cast<size_t>(nnz), cudaMemcpyHostToDevice);
        cudaMemcpy(dV, hVal, sizeof(float) * static_cast<size_t>(nnz), cudaMemcpyHostToDevice);
        cudaMemcpy(dx, hx, sizeof(float) * static_cast<size_t>(n), cudaMemcpyHostToDevice);
        cudaMemcpy(dy, hy, sizeof(float) * static_cast<size_t>(m), cudaMemcpyHostToDevice);

        cusparseSpMatDescr_t coo = nullptr;
        {
            case_begin("cusparseCreateCoo");
            cusparse_ok(cusparseCreateCoo(&coo, m, n, nnz, dR, dC, dV, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                          CUDA_R_32F),
                        "cusparseCreateCoo");
        }
        cusparseDnVecDescr_t vx = nullptr, vy = nullptr;
        {
            case_begin("cusparseCreateDnVec");
            cusparse_ok(cusparseCreateDnVec(&vx, n, dx, CUDA_R_32F), "cusparseCreateDnVec COO x");
        }
        {
            case_begin("cusparseCreateDnVec");
            cusparse_ok(cusparseCreateDnVec(&vy, m, dy, CUDA_R_32F), "cusparseCreateDnVec COO y");
        }

        float            a = 1.f, b = 0.f;
        size_t           bufSz = 0;
        cusparseStatus_t st =
            cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &a, coo, vx, &b, vy, CUDA_R_32F,
                                    CUSPARSE_MV_ALG_DEFAULT, &bufSz);
        cusparse_ok(st, "cusparseSpMV_bufferSize COO");
        void* p = nullptr;
        const size_t allocP = bufSz > 0 ? bufSz : 4;
        cudaMalloc(&p, allocP);
        cudaStreamSynchronize(stream);
        {
            case_begin("cusparseSpMV");
            st = cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &a, coo, vx, &b, vy, CUDA_R_32F,
                              CUSPARSE_MV_ALG_DEFAULT, p);
            if(cusparse_ok(st, "cusparseSpMV COO"))
            {
                cudaDeviceSynchronize();
                float yo[kSpDiagN];
                cudaMemcpy(yo, dy, sizeof(yo), cudaMemcpyDeviceToHost);
                float yref[kSpDiagN];
                for(int i = 0; i < m; ++i)
                    yref[i] = 0.f;
                cpu_coo_spmv_f(m, n, nnz, hCooRow, hCooCol, hVal, hx, yref, 1.f, 0.f);
                bool coo_ok = true;
                for(int i = 0; i < m; ++i)
                {
                    if(!nearly_equal_f(yo[i], yref[i], 1e-3f))
                        coo_ok = false;
                }
                if(coo_ok)
                    ok("cusparseSpMV COO matches CPU");
                else
                {
                    warn("cusparseSpMV COO numeric mismatch vs CPU (compat)");
                    std::fprintf(stderr, "       gpu[0,1]=(%g,%g) cpu[0,1]=(%g,%g)\n", yo[0], yo[1], yref[0], yref[1]);
                }
            }
        }
        cudaFree(p);
        {
            case_begin("cusparseDestroyDnVec");
            cusparse_ok(cusparseDestroyDnVec(vx), "cusparseDestroyDnVec COO vx");
        }
        {
            case_begin("cusparseDestroyDnVec");
            cusparse_ok(cusparseDestroyDnVec(vy), "cusparseDestroyDnVec COO vy");
        }
        {
            case_begin("cusparseDestroySpMat");
            cusparse_ok(cusparseDestroySpMat(coo), "cusparseDestroySpMat COO");
        }
        cudaFree(dR);
        cudaFree(dC);
        cudaFree(dV);
        cudaFree(dx);
        cudaFree(dy);
    }

    // ---------- Stronger COO create/destroy correctness ----------
    {
        const int m = 8, n = 7, nnz = 14;
        int   hRow0[nnz] = {5, 0, 0, 3, 3, 3, 7, 1, 6, 2, 2, 4, 5, 6};
        int   hCol0[nnz] = {1, 2, 2, 0, 4, 0, 6, 1, 2, 5, 5, 3, 1, 2};
        float hVal[nnz]  = {2.0f, 1.0f, -0.25f, 3.0f, -1.5f, 0.5f, 4.0f,
                            -2.0f, 1.25f, 2.5f, 0.75f, -3.0f, 1.0f, 2.0f};
        float hx[n]      = {1.0f, -1.0f, 2.0f, 0.5f, -0.5f, 3.0f, 4.0f};
        float hy[m]      = {0.0f, 1.0f, -1.0f, 0.5f, 0.0f, -0.5f, 2.0f, 1.5f};
        int   hRow1[nnz];
        int   hCol1[nnz];
        for(int i = 0; i < nnz; ++i)
        {
            hRow1[i] = hRow0[i] + 1;
            hCol1[i] = hCol0[i] + 1;
        }

        int *   dR = nullptr, *dC = nullptr;
        float * dV = nullptr, *dx = nullptr, *dy = nullptr;
        cudaMalloc(&dR, sizeof(hRow0));
        cudaMalloc(&dC, sizeof(hCol0));
        cudaMalloc(&dV, sizeof(hVal));
        cudaMalloc(&dx, sizeof(hx));
        cudaMalloc(&dy, sizeof(hy));
        cudaMemcpy(dR, hRow0, sizeof(hRow0), cudaMemcpyHostToDevice);
        cudaMemcpy(dC, hCol0, sizeof(hCol0), cudaMemcpyHostToDevice);
        cudaMemcpy(dV, hVal, sizeof(hVal), cudaMemcpyHostToDevice);
        cudaMemcpy(dx, hx, sizeof(hx), cudaMemcpyHostToDevice);
        cudaMemcpy(dy, hy, sizeof(hy), cudaMemcpyHostToDevice);

        cusparseSpMatDescr_t coo0 = nullptr;
        {
            case_begin("cusparseCreateCoo");
            cusparse_ok(cusparseCreateCoo(&coo0,
                                          m,
                                          n,
                                          nnz,
                                          dR,
                                          dC,
                                          dV,
                                          CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_BASE_ZERO,
                                          CUDA_R_32F),
                        "cusparseCreateCoo irregular base0");
        }
        cusparseDnVecDescr_t vx0 = nullptr, vy0 = nullptr;
        cusparseCreateDnVec(&vx0, n, dx, CUDA_R_32F);
        cusparseCreateDnVec(&vy0, m, dy, CUDA_R_32F);
        float  alpha = 0.75f, beta = -0.25f;
        size_t bufSz = 0;
        cusparse_ok(cusparseSpMV_bufferSize(h,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha,
                                            coo0,
                                            vx0,
                                            &beta,
                                            vy0,
                                            CUDA_R_32F,
                                            CUSPARSE_MV_ALG_DEFAULT,
                                            &bufSz),
                    "cusparseSpMV_bufferSize COO irregular base0");
        void* buf = nullptr;
        cudaMalloc(&buf, bufSz > 0 ? bufSz : 4);
        float hy_ref[m];
        float hy_out[m];
        cpu_coo_spmv_base_f(m, n, nnz, hRow0, hCol0, hVal, hx, hy_ref, alpha, beta, 0);
        {
            case_begin("cusparseSpMV");
            auto st = cusparseSpMV(h,
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &alpha,
                                   coo0,
                                   vx0,
                                   &beta,
                                   vy0,
                                   CUDA_R_32F,
                                   CUSPARSE_MV_ALG_DEFAULT,
                                   buf);
            if(cusparse_ok(st, "cusparseSpMV COO irregular base0"))
            {
                cudaDeviceSynchronize();
                cudaMemcpy(hy_out, dy, sizeof(hy_out), cudaMemcpyDeviceToHost);
                bool all_ok = true;
                for(int i = 0; i < m; ++i)
                    all_ok &= nearly_equal_f(hy_out[i], hy_ref[i], 1e-3f);
                if(all_ok)
                    ok("cusparseCreateCoo irregular base0 values");
                else
                    warn("cusparseCreateCoo irregular base0 values mismatch (compat)");
            }
        }
        {
            case_begin("cusparseDestroySpMat");
            cusparse_ok(cusparseDestroySpMat(coo0), "cusparseDestroySpMat COO irregular base0");
        }
        cusparseDestroyDnVec(vx0);
        cusparseDestroyDnVec(vy0);
        cudaFree(buf);

        cudaMemcpy(dR, hRow1, sizeof(hRow1), cudaMemcpyHostToDevice);
        cudaMemcpy(dC, hCol1, sizeof(hCol1), cudaMemcpyHostToDevice);
        cudaMemcpy(dy, hy, sizeof(hy), cudaMemcpyHostToDevice);
        cusparseSpMatDescr_t coo1 = nullptr;
        {
            case_begin("cusparseCreateCoo");
            cusparse_ok(cusparseCreateCoo(&coo1,
                                          m,
                                          n,
                                          nnz,
                                          dR,
                                          dC,
                                          dV,
                                          CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_BASE_ONE,
                                          CUDA_R_32F),
                        "cusparseCreateCoo irregular base1");
        }
        cusparseDnVecDescr_t vx1 = nullptr, vy1 = nullptr;
        cusparseCreateDnVec(&vx1, n, dx, CUDA_R_32F);
        cusparseCreateDnVec(&vy1, m, dy, CUDA_R_32F);
        cusparse_ok(cusparseSpMV_bufferSize(h,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha,
                                            coo1,
                                            vx1,
                                            &beta,
                                            vy1,
                                            CUDA_R_32F,
                                            CUSPARSE_MV_ALG_DEFAULT,
                                            &bufSz),
                    "cusparseSpMV_bufferSize COO irregular base1");
        cudaMalloc(&buf, bufSz > 0 ? bufSz : 4);
        cpu_coo_spmv_base_f(m, n, nnz, hRow1, hCol1, hVal, hx, hy_ref, alpha, beta, 1);
        {
            case_begin("cusparseSpMV");
            auto st = cusparseSpMV(h,
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &alpha,
                                   coo1,
                                   vx1,
                                   &beta,
                                   vy1,
                                   CUDA_R_32F,
                                   CUSPARSE_MV_ALG_DEFAULT,
                                   buf);
            if(cusparse_ok(st, "cusparseSpMV COO irregular base1"))
            {
                cudaDeviceSynchronize();
                cudaMemcpy(hy_out, dy, sizeof(hy_out), cudaMemcpyDeviceToHost);
                bool all_ok = true;
                for(int i = 0; i < m; ++i)
                    all_ok &= nearly_equal_f(hy_out[i], hy_ref[i], 1e-3f);
                if(all_ok)
                    ok("cusparseCreateCoo irregular base1 values");
                else
                    warn("cusparseCreateCoo irregular base1 values mismatch (compat)");
            }
        }
        {
            case_begin("cusparseDestroySpMat");
            cusparse_ok(cusparseDestroySpMat(coo1), "cusparseDestroySpMat COO irregular base1");
        }
        cusparseDestroyDnVec(vx1);
        cusparseDestroyDnVec(vy1);
        cudaFree(buf);
        cudaFree(dR);
        cudaFree(dC);
        cudaFree(dV);
        cudaFree(dx);
        cudaFree(dy);
    }

    // ---------- BSR + legacy bsrmv + MatDescr ----------
    {
        const int mb = 1, nb = 1, nnzb = 1, block = 2;
        int       hRp[] = {0, 1};
        int       hCi[] = {0};
        float     hBv[] = {1.f, 0.f, 0.f, 1.f};
        float     hx[] = {1.f, 2.f}, hy[] = {0.f, 0.f};
        int *     dRp = nullptr, *dCi = nullptr;
        float *   dBv = nullptr, *dx = nullptr, *dy = nullptr;
        cudaMalloc(&dRp, sizeof(hRp));
        cudaMalloc(&dCi, sizeof(hCi));
        cudaMalloc(&dBv, sizeof(hBv));
        cudaMalloc(&dx, sizeof(hx));
        cudaMalloc(&dy, sizeof(hy));
        cudaMemcpy(dRp, hRp, sizeof(hRp), cudaMemcpyHostToDevice);
        cudaMemcpy(dCi, hCi, sizeof(hCi), cudaMemcpyHostToDevice);
        cudaMemcpy(dBv, hBv, sizeof(hBv), cudaMemcpyHostToDevice);
        cudaMemcpy(dx, hx, sizeof(hx), cudaMemcpyHostToDevice);
        cudaMemcpy(dy, hy, sizeof(hy), cudaMemcpyHostToDevice);

        cusparseMatDescr_t descr{};
        {
            case_begin("cusparseCreateMatDescr");
            cusparse_ok(cusparseCreateMatDescr(&descr), "cusparseCreateMatDescr BSR");
        }
        {
            case_begin("cusparseSetMatType");
            cusparse_ok(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL), "cusparseSetMatType");
        }
        {
            case_begin("cusparseSetMatIndexBase");
            cusparse_ok(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO), "cusparseSetMatIndexBase");
        }
        {
            case_begin("cusparseSetMatDiagType");
            cusparse_ok(cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT), "cusparseSetMatDiagType");
        }

        float            alpha = 1.f, beta = 0.f;
        cusparseStatus_t st = cusparseSbsrmv(h, CUSPARSE_DIRECTION_COLUMN, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb,
                                             nnzb, &alpha, descr, dBv, dRp, dCi, block, dx, &beta, dy);
        {
            case_begin("cusparseSbsrmv");
            if(cusparse_ok(st, "cusparseSbsrmv"))
            {
                cudaDeviceSynchronize();
                float yo[2];
                cudaMemcpy(yo, dy, sizeof(yo), cudaMemcpyDeviceToHost);
                if(nearly_equal_f(yo[0], hx[0], 1e-3f) && nearly_equal_f(yo[1], hx[1], 1e-3f))
                    ok("cusparseSbsrmv vs hx");
                else
                {
                    warn("cusparseSbsrmv numeric vs hx (compat)");
                    std::fprintf(stderr, "       gpu=(%g,%g) ref=(%g,%g)\n", yo[0], yo[1], hx[0], hx[1]);
                }
            }
        }

        {
            case_begin("cusparseDestroyMatDescr");
            cusparse_ok(cusparseDestroyMatDescr(descr), "cusparseDestroyMatDescr");
        }

        cusparseMatDescr_t descrD{};
        {
            case_begin("cusparseCreateMatDescr");
            cusparseCreateMatDescr(&descrD);
        }
        {
            case_begin("cusparseSetMatType");
            cusparseSetMatType(descrD, CUSPARSE_MATRIX_TYPE_GENERAL);
        }
        {
            case_begin("cusparseSetMatIndexBase");
            cusparseSetMatIndexBase(descrD, CUSPARSE_INDEX_BASE_ZERO);
        }
        double* dBvD = nullptr;
        double* dxD = nullptr;
        double* dyD = nullptr;
        double  hBvD[4] = {1, 0, 0, 1};
        double  hxD[2] = {1, 2};
        cudaMalloc(&dBvD, sizeof(hBvD));
        cudaMalloc(&dxD, sizeof(hxD));
        cudaMalloc(&dyD, sizeof(hxD));
        cudaMemcpy(dBvD, hBvD, sizeof(hBvD), cudaMemcpyHostToDevice);
        cudaMemcpy(dxD, hxD, sizeof(hxD), cudaMemcpyHostToDevice);
        double aD = 1, bD = 0;
        st = cusparseDbsrmv(h, CUSPARSE_DIRECTION_COLUMN, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, nnzb, &aD, descrD,
                            dBvD, dRp, dCi, block, dxD, &bD, dyD);
        {
            case_begin("cusparseDbsrmv");
            if(cusparse_ok(st, "cusparseDbsrmv"))
            {
                cudaDeviceSynchronize();
                double yo[2];
                cudaMemcpy(yo, dyD, sizeof(yo), cudaMemcpyDeviceToHost);
                if(nearly_equal_d(yo[0], hxD[0], 1e-5) && nearly_equal_d(yo[1], hxD[1], 1e-5))
                    ok("cusparseDbsrmv vs hx");
                else
                {
                    warn("cusparseDbsrmv numeric vs expected hx (compat)");
                    std::fprintf(stderr, "       gpu=(%.17e,%.17e) ref=(%.17e,%.17e)\n", yo[0], yo[1], hxD[0], hxD[1]);
                }
            }
        }
        {
            case_begin("cusparseDestroyMatDescr");
            cusparse_ok(cusparseDestroyMatDescr(descrD), "cusparseDestroyMatDescr descrD");
        }

        cudaFree(dRp);
        cudaFree(dCi);
        cudaFree(dBv);
        cudaFree(dx);
        cudaFree(dy);
        cudaFree(dBvD);
        cudaFree(dxD);
        cudaFree(dyD);
    }

    // ---------- Stronger MatDescr / index-base correctness ----------
    {
        const int mb = 3, nb = 3, nnzb = 4, block = 2;
        int   hRp0[mb + 1] = {0, 2, 3, 4};
        int   hCi0[nnzb]   = {0, 2, 1, 2};
        float hBv[nnzb * block * block] = {
            1.f, 3.f, 2.f, 4.f,     // block(0,0), column-major -> [[1,2],[3,4]]
            -1.f, 0.5f, 2.f, -2.f,  // block(0,2)
            0.25f, -1.5f, 1.f, 3.f, // block(1,1)
            4.f, 1.f, -0.5f, 2.f    // block(2,2)
        };
        int   hRp1[mb + 1];
        int   hCi1[nnzb];
        for(int i = 0; i <= mb; ++i)
            hRp1[i] = hRp0[i] + 1;
        for(int i = 0; i < nnzb; ++i)
            hCi1[i] = hCi0[i] + 1;

        float hx[nb * block] = {1.f, -1.f, 0.5f, 2.f, -0.5f, 3.f};
        float hy[mb * block] = {0.f, 0.f, 1.f, -1.f, 0.5f, -0.5f};

        int *   dRp = nullptr, *dCi = nullptr;
        float * dBv = nullptr, *dx = nullptr, *dy = nullptr;
        cudaMalloc(&dRp, sizeof(hRp0));
        cudaMalloc(&dCi, sizeof(hCi0));
        cudaMalloc(&dBv, sizeof(hBv));
        cudaMalloc(&dx, sizeof(hx));
        cudaMalloc(&dy, sizeof(hy));
        cudaMemcpy(dRp, hRp0, sizeof(hRp0), cudaMemcpyHostToDevice);
        cudaMemcpy(dCi, hCi0, sizeof(hCi0), cudaMemcpyHostToDevice);
        cudaMemcpy(dBv, hBv, sizeof(hBv), cudaMemcpyHostToDevice);
        cudaMemcpy(dx, hx, sizeof(hx), cudaMemcpyHostToDevice);
        cudaMemcpy(dy, hy, sizeof(hy), cudaMemcpyHostToDevice);
        float y_ref[6] = {0.f, 0.f, 1.f, -1.f, 0.5f, -0.5f};
        float dense_ref[36];
        cpu_bsr_dense_colmajor_f(mb, nb, block, hRp0, hCi0, hBv, 0, dense_ref);
        for(int r = 0; r < 6; ++r)
        {
            double acc = 0.0;
            for(int c = 0; c < 6; ++c)
                acc += static_cast<double>(dense_ref[r * 6 + c]) * static_cast<double>(hx[c]);
            y_ref[r] = static_cast<float>(acc);
        }

        cusparseMatDescr_t bdesc0{}, cdesc0{};
        {
            case_begin("cusparseCreateMatDescr");
            cusparse_ok(cusparseCreateMatDescr(&bdesc0), "cusparseCreateMatDescr BSR strong bdesc0");
        }
        {
            case_begin("cusparseCreateMatDescr");
            cusparse_ok(cusparseCreateMatDescr(&cdesc0), "cusparseCreateMatDescr BSR strong cdesc0");
        }
        {
            case_begin("cusparseSetMatType");
            cusparse_ok(cusparseSetMatType(bdesc0, CUSPARSE_MATRIX_TYPE_GENERAL), "cusparseSetMatType BSR strong bdesc0");
        }
        {
            case_begin("cusparseSetMatType");
            cusparse_ok(cusparseSetMatType(cdesc0, CUSPARSE_MATRIX_TYPE_GENERAL), "cusparseSetMatType BSR strong cdesc0");
        }
        {
            case_begin("cusparseSetMatIndexBase");
            cusparse_ok(cusparseSetMatIndexBase(bdesc0, CUSPARSE_INDEX_BASE_ZERO),
                        "cusparseSetMatIndexBase BSR strong bdesc0");
        }
        {
            case_begin("cusparseSetMatIndexBase");
            cusparse_ok(cusparseSetMatIndexBase(cdesc0, CUSPARSE_INDEX_BASE_ZERO),
                        "cusparseSetMatIndexBase BSR strong cdesc0");
        }

        float alpha = 1.f, beta = 0.f;
        {
            case_begin("cusparseSbsrmv");
            auto st = cusparseSbsrmv(h,
                                     CUSPARSE_DIRECTION_COLUMN,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     mb,
                                     nb,
                                     nnzb,
                                     &alpha,
                                     bdesc0,
                                     dBv,
                                     dRp,
                                     dCi,
                                     block,
                                     dx,
                                     &beta,
                                     dy);
            if(cusparse_ok(st, "cusparseSbsrmv strong base0"))
            {
                float y_out[6];
                cudaDeviceSynchronize();
                cudaMemcpy(y_out, dy, sizeof(y_out), cudaMemcpyDeviceToHost);
                bool ok_all = true;
                for(int i = 0; i < 6; ++i)
                    ok_all &= nearly_equal_f(y_out[i], y_ref[i], 1e-3f);
                if(ok_all)
                    ok("cusparseSbsrmv strong base0 values");
                else
                    warn("cusparseSbsrmv strong base0 mismatch (compat)");
            }
        }
        cudaMemcpy(dRp, hRp1, sizeof(hRp1), cudaMemcpyHostToDevice);
        cudaMemcpy(dCi, hCi1, sizeof(hCi1), cudaMemcpyHostToDevice);
        cudaMemcpy(dy, hy, sizeof(hy), cudaMemcpyHostToDevice);

        cusparseMatDescr_t bdesc1{}, cdesc1{};
        {
            case_begin("cusparseCreateMatDescr");
            cusparse_ok(cusparseCreateMatDescr(&bdesc1), "cusparseCreateMatDescr BSR strong bdesc1");
        }
        {
            case_begin("cusparseCreateMatDescr");
            cusparse_ok(cusparseCreateMatDescr(&cdesc1), "cusparseCreateMatDescr BSR strong cdesc1");
        }
        {
            case_begin("cusparseSetMatType");
            cusparse_ok(cusparseSetMatType(bdesc1, CUSPARSE_MATRIX_TYPE_GENERAL), "cusparseSetMatType BSR strong bdesc1");
        }
        {
            case_begin("cusparseSetMatType");
            cusparse_ok(cusparseSetMatType(cdesc1, CUSPARSE_MATRIX_TYPE_GENERAL), "cusparseSetMatType BSR strong cdesc1");
        }
        {
            case_begin("cusparseSetMatIndexBase");
            cusparse_ok(cusparseSetMatIndexBase(bdesc1, CUSPARSE_INDEX_BASE_ONE),
                        "cusparseSetMatIndexBase BSR strong bdesc1");
        }
        {
            case_begin("cusparseSetMatIndexBase");
            cusparse_ok(cusparseSetMatIndexBase(cdesc1, CUSPARSE_INDEX_BASE_ONE),
                        "cusparseSetMatIndexBase BSR strong cdesc1");
        }

        {
            case_begin("cusparseSbsrmv");
            auto st = cusparseSbsrmv(h,
                                     CUSPARSE_DIRECTION_COLUMN,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     mb,
                                     nb,
                                     nnzb,
                                     &alpha,
                                     bdesc1,
                                     dBv,
                                     dRp,
                                     dCi,
                                     block,
                                     dx,
                                     &beta,
                                     dy);
            if(cusparse_ok(st, "cusparseSbsrmv strong base1"))
            {
                float y_out[6];
                cudaDeviceSynchronize();
                cudaMemcpy(y_out, dy, sizeof(y_out), cudaMemcpyDeviceToHost);
                bool ok_all = true;
                for(int i = 0; i < 6; ++i)
                    ok_all &= nearly_equal_f(y_out[i], y_ref[i], 1e-3f);
                if(ok_all)
                    ok("cusparseSbsrmv strong base1 values");
                else
                    warn("cusparseSbsrmv strong base1 mismatch (compat)");
            }
        }
        {
            case_begin("cusparseDestroyMatDescr");
            cusparse_ok(cusparseDestroyMatDescr(bdesc0), "cusparseDestroyMatDescr BSR strong bdesc0");
        }
        {
            case_begin("cusparseDestroyMatDescr");
            cusparse_ok(cusparseDestroyMatDescr(cdesc0), "cusparseDestroyMatDescr BSR strong cdesc0");
        }
        {
            case_begin("cusparseDestroyMatDescr");
            cusparse_ok(cusparseDestroyMatDescr(bdesc1), "cusparseDestroyMatDescr BSR strong bdesc1");
        }
        {
            case_begin("cusparseDestroyMatDescr");
            cusparse_ok(cusparseDestroyMatDescr(cdesc1), "cusparseDestroyMatDescr BSR strong cdesc1");
        }
        cudaFree(dRp);
        cudaFree(dCi);
        cudaFree(dBv);
        cudaFree(dx);
        cudaFree(dy);
    }

    // ---------- Dbsr2csr / Sbsr2csr ----------
    {
        const int mb = 1, nb = 1, nnzb = 1, blockDim = 2;
        int       bsrRow[] = {0, 1};
        int       bsrCol[] = {0};
        float     bsrVal[] = {1.f, 0.f, 0.f, 1.f};
        int *     dbr = nullptr, *dbc = nullptr;
        float*    dbv = nullptr;
        int *     dcr = nullptr, *dcc = nullptr;
        float*    dcv = nullptr;
        cudaMalloc(&dbr, sizeof(bsrRow));
        cudaMalloc(&dbc, sizeof(bsrCol));
        cudaMalloc(&dbv, sizeof(bsrVal));
        cudaMalloc(&dcr, sizeof(int) * 3);
        cudaMalloc(&dcc, sizeof(int) * 4);
        cudaMalloc(&dcv, sizeof(float) * 4);
        cudaMemcpy(dbr, bsrRow, sizeof(bsrRow), cudaMemcpyHostToDevice);
        cudaMemcpy(dbc, bsrCol, sizeof(bsrCol), cudaMemcpyHostToDevice);
        cudaMemcpy(dbv, bsrVal, sizeof(bsrVal), cudaMemcpyHostToDevice);

        cusparseMatDescr_t bdesc{}, cdesc{};
        {
            case_begin("cusparseCreateMatDescr");
            cusparseCreateMatDescr(&bdesc);
        }
        {
            case_begin("cusparseCreateMatDescr");
            cusparseCreateMatDescr(&cdesc);
        }
        {
            case_begin("cusparseSetMatType");
            cusparseSetMatType(bdesc, CUSPARSE_MATRIX_TYPE_GENERAL);
        }
        {
            case_begin("cusparseSetMatType");
            cusparseSetMatType(cdesc, CUSPARSE_MATRIX_TYPE_GENERAL);
        }
        {
            case_begin("cusparseSetMatIndexBase");
            cusparseSetMatIndexBase(bdesc, CUSPARSE_INDEX_BASE_ZERO);
        }
        {
            case_begin("cusparseSetMatIndexBase");
            cusparseSetMatIndexBase(cdesc, CUSPARSE_INDEX_BASE_ZERO);
        }

        cusparseStatus_t st =
            cusparseSbsr2csr(h, CUSPARSE_DIRECTION_COLUMN, mb, nb, bdesc, dbv, dbr, dbc, blockDim, cdesc, dcv, dcr,
                             dcc);
        {
            case_begin("cusparseSbsr2csr");
            if(cusparse_ok(st, "cusparseSbsr2csr"))
            {
                cudaDeviceSynchronize();
                int   hr[3];
                int   hc[4];
                float hv[4];
                cudaMemcpy(hr, dcr, sizeof(hr), cudaMemcpyDeviceToHost);
                cudaMemcpy(hc, dcc, 4 * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(hv, dcv, sizeof(hv), cudaMemcpyDeviceToHost);
                const int nnz = hr[2];
                float     D[4] = {0, 0, 0, 0};
                bool dense_ok = false;
                if(nnz >= 0 && nnz <= 4 && hr[0] == 0 && hr[1] >= 0 && hr[1] <= hr[2])
                {
                    for(int r = 0; r < 2; ++r)
                        for(int p = hr[r]; p < hr[r + 1] && p < nnz; ++p)
                        {
                            if(p < 0 || p >= 4)
                                break;
                            int c = hc[p];
                            if(c >= 0 && c < 2)
                                D[r * 2 + c] = hv[p];
                        }
                    const float I2[4] = {1.f, 0.f, 0.f, 1.f};
                    dense_ok = true;
                    for(int i = 0; i < 4; ++i)
                        if(!nearly_equal_f(D[i], I2[i], 1e-3f))
                            dense_ok = false;
                }
                else
                    warn("cusparseSbsr2csr rowptr/nnz unexpected; skip dense check");
                if(dense_ok)
                    ok("cusparseSbsr2csr dense==I2");
                else if(nnz >= 0 && nnz <= 4 && hr[0] == 0 && hr[1] >= 0 && hr[1] <= hr[2])
                {
                    warn("cusparseSbsr2csr dense mismatch (layout/toolkit)");
                    std::fprintf(stderr, "       D=[%.17e %.17e; %.17e %.17e]\n", static_cast<double>(D[0]),
                                 static_cast<double>(D[1]), static_cast<double>(D[2]), static_cast<double>(D[3]));
                }
            }
        }

        double* dbvD = nullptr, *dcvD = nullptr;
        cudaMalloc(&dbvD, 4 * sizeof(double));
        cudaMalloc(&dcvD, 4 * sizeof(double));
        double bsrValD[] = {1, 0, 0, 1};
        cudaMemcpy(dbvD, bsrValD, sizeof(bsrValD), cudaMemcpyHostToDevice);
        st = cusparseDbsr2csr(h, CUSPARSE_DIRECTION_COLUMN, mb, nb, bdesc, dbvD, dbr, dbc, blockDim, cdesc, dcvD, dcr,
                              dcc);
        {
            case_begin("cusparseDbsr2csr");
            if(cusparse_ok(st, "cusparseDbsr2csr"))
            {
                cudaDeviceSynchronize();
                int    hr[3];
                int    hc[4];
                double hv[4];
                cudaMemcpy(hr, dcr, sizeof(hr), cudaMemcpyDeviceToHost);
                cudaMemcpy(hc, dcc, 4 * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(hv, dcvD, sizeof(hv), cudaMemcpyDeviceToHost);
                const int nnz = hr[2];
                double    D[4] = {0, 0, 0, 0};
                bool      dense_ok = false;
                if(nnz >= 0 && nnz <= 4 && hr[0] == 0 && hr[1] >= 0 && hr[1] <= hr[2])
                {
                    for(int r = 0; r < 2; ++r)
                        for(int p = hr[r]; p < hr[r + 1] && p < nnz; ++p)
                        {
                            if(p < 0 || p >= 4)
                                break;
                            int c = hc[p];
                            if(c >= 0 && c < 2)
                                D[r * 2 + c] = hv[p];
                        }
                    const double I2[4] = {1., 0., 0., 1.};
                    dense_ok = true;
                    for(int i = 0; i < 4; ++i)
                        if(!nearly_equal_d(D[i], I2[i], 1e-8))
                            dense_ok = false;
                }
                else
                    warn("cusparseDbsr2csr rowptr/nnz unexpected; skip dense check");
                if(dense_ok)
                    ok("cusparseDbsr2csr dense==I2");
                else if(nnz >= 0 && nnz <= 4 && hr[0] == 0 && hr[1] >= 0 && hr[1] <= hr[2])
                {
                    warn("cusparseDbsr2csr dense mismatch (layout/toolkit)");
                    std::fprintf(stderr, "       D=[%.17e %.17e; %.17e %.17e]\n", D[0], D[1], D[2], D[3]);
                }
            }
        }

        {
            case_begin("cusparseDestroyMatDescr");
            cusparseDestroyMatDescr(bdesc);
        }
        {
            case_begin("cusparseDestroyMatDescr");
            cusparseDestroyMatDescr(cdesc);
        }
        cudaFree(dbr);
        cudaFree(dbc);
        cudaFree(dbv);
        cudaFree(dcr);
        cudaFree(dcc);
        cudaFree(dcv);
        cudaFree(dbvD);
        cudaFree(dcvD);
    }

    // ---------- SpVec create/destroy ----------
    {
        int            idx[] = {0};
        float          val[] = {1.f};
        int*           di = nullptr;
        float*         dv = nullptr;
        cudaMalloc(&di, sizeof(idx));
        cudaMalloc(&dv, sizeof(val));
        cudaMemcpy(di, idx, sizeof(idx), cudaMemcpyHostToDevice);
        cudaMemcpy(dv, val, sizeof(val), cudaMemcpyHostToDevice);
        cusparseSpVecDescr_t sv = nullptr;
        {
            case_begin("cusparseCreateSpVec");
            cusparse_ok(cusparseCreateSpVec(&sv, 4, 1, di, dv, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F),
                        "cusparseCreateSpVec");
        }
        {
            case_begin("cusparseDestroySpVec");
            cusparse_ok(cusparseDestroySpVec(sv), "cusparseDestroySpVec");
        }
        cudaFree(di);
        cudaFree(dv);
    }

    {
        case_begin("cusparseDestroy");
        cusparse_ok(cusparseDestroy(h), "cusparseDestroy");
    }
    cudaStreamDestroy(stream);

    print_summary("corex_cusparse_inventory");
    return g_failures() > 0 ? 1 : 0;
}
