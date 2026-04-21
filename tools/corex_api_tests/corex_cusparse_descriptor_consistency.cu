#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace
{
bool nearf(float a, float b, float eps = 1e-6f)
{
    return std::fabs(a - b) <= eps * (1.f + std::max(std::fabs(a), std::fabs(b)));
}

bool check_csr_rules(int m, int n, int nnz, const int* rowptr, const int* colidx, int base)
{
    if(rowptr[0] != base || rowptr[m] != nnz + base)
        return false;
    for(int i = 0; i < m; ++i)
        if(rowptr[i] > rowptr[i + 1])
            return false;
    for(int i = 0; i < m; ++i)
    {
        int rb = rowptr[i] - base;
        int re = rowptr[i + 1] - base;
        for(int p = rb; p < re; ++p)
        {
            int c = colidx[p] - base;
            if(c < 0 || c >= n)
                return false;
        }
    }
    return true;
}

bool check_coo_rules(int m, int n, int nnz, const int* row, const int* col, int base)
{
    for(int p = 0; p < nnz; ++p)
    {
        int r = row[p] - base;
        int c = col[p] - base;
        if(r < 0 || r >= m || c < 0 || c >= n)
            return false;
        if(p > 0 && row[p - 1] > row[p])
            return false;  // row-sorted requirement
    }
    return true;
}

void print_status(const char* tag, bool ok)
{
    std::printf("[%s] %s\n", tag, ok ? "PASS" : "FAIL");
}
}  // namespace

int main()
{
    cudaSetDevice(0);
    cudaStream_t stream{};
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cusparseHandle_t h{};
    cusparseCreate(&h);
    cusparseSetStream(h, stream);

    // ---------------- CSR descriptor consistency ----------------
    {
        constexpr int m = 4, n = 5, nnz = 8;
        int   hRow[m + 1] = {0, 2, 4, 7, 8};
        int   hCol[nnz]   = {0, 3, 1, 4, 0, 2, 4, 3};
        float hVal[nnz]   = {2.f, -1.f, 3.f, 0.5f, -2.f, 1.25f, 4.f, -0.75f};

        print_status("CSR input rule check", check_csr_rules(m, n, nnz, hRow, hCol, 0));

        int *  dRow = nullptr, *dCol = nullptr;
        float* dVal = nullptr;
        cudaMalloc(&dRow, sizeof(hRow));
        cudaMalloc(&dCol, sizeof(hCol));
        cudaMalloc(&dVal, sizeof(hVal));
        cudaMemcpy(dRow, hRow, sizeof(hRow), cudaMemcpyHostToDevice);
        cudaMemcpy(dCol, hCol, sizeof(hCol), cudaMemcpyHostToDevice);
        cudaMemcpy(dVal, hVal, sizeof(hVal), cudaMemcpyHostToDevice);

        cusparseSpMatDescr_t A = nullptr;
        auto                 st =
            cusparseCreateCsr(&A,
                              m,
                              n,
                              nnz,
                              dRow,
                              dCol,
                              dVal,
                              CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_BASE_ZERO,
                              CUDA_R_32F);
        print_status("CSR create status", st == CUSPARSE_STATUS_SUCCESS);

        cusparseFormat_t fmt{};
        st = cusparseSpMatGetFormat(A, &fmt);
        print_status("CSR get format", st == CUSPARSE_STATUS_SUCCESS && fmt == CUSPARSE_FORMAT_CSR);

        cusparseIndexBase_t base{};
        st = cusparseSpMatGetIndexBase(A, &base);
        print_status("CSR get base", st == CUSPARSE_STATUS_SUCCESS && base == CUSPARSE_INDEX_BASE_ZERO);

        int64_t rows = -1, cols = -1, got_nnz = -1;
        void *  pRow = nullptr, *pCol = nullptr, *pVal = nullptr;
        cusparseIndexType_t itRow{}, itCol{};
        cudaDataType        dt{};
        st = cusparseCsrGet(A, &rows, &cols, &got_nnz, &pRow, &pCol, &pVal, &itRow, &itCol, &base, &dt);
        print_status("CSR get meta",
                     st == CUSPARSE_STATUS_SUCCESS && rows == m && cols == n && got_nnz == nnz
                         && itRow == CUSPARSE_INDEX_32I && itCol == CUSPARSE_INDEX_32I && dt == CUDA_R_32F);
        print_status("CSR pointer identity", pRow == dRow && pCol == dCol && pVal == dVal);

        int   row_back[m + 1];
        int   col_back[nnz];
        float val_back[nnz];
        cudaMemcpy(row_back, static_cast<int*>(pRow), sizeof(row_back), cudaMemcpyDeviceToHost);
        cudaMemcpy(col_back, static_cast<int*>(pCol), sizeof(col_back), cudaMemcpyDeviceToHost);
        cudaMemcpy(val_back, static_cast<float*>(pVal), sizeof(val_back), cudaMemcpyDeviceToHost);

        bool arrays_ok = std::memcmp(row_back, hRow, sizeof(hRow)) == 0
                         && std::memcmp(col_back, hCol, sizeof(hCol)) == 0;
        for(int i = 0; i < nnz && arrays_ok; ++i)
            arrays_ok = nearf(val_back[i], hVal[i]);
        print_status("CSR array roundtrip", arrays_ok);
        print_status("CSR roundtrip rule check", check_csr_rules(m, n, nnz, row_back, col_back, 0));

        cusparseDestroySpMat(A);
        cudaFree(dRow);
        cudaFree(dCol);
        cudaFree(dVal);

        // negative CSR input
        int bad_row[m + 1] = {0, 2, 4, 7, 7};
        print_status("CSR malformed input rule check", !check_csr_rules(m, n, nnz, bad_row, hCol, 0));
        cudaMalloc(&dRow, sizeof(bad_row));
        cudaMalloc(&dCol, sizeof(hCol));
        cudaMalloc(&dVal, sizeof(hVal));
        cudaMemcpy(dRow, bad_row, sizeof(bad_row), cudaMemcpyHostToDevice);
        cudaMemcpy(dCol, hCol, sizeof(hCol), cudaMemcpyHostToDevice);
        cudaMemcpy(dVal, hVal, sizeof(hVal), cudaMemcpyHostToDevice);
        st = cusparseCreateCsr(&A,
                               m,
                               n,
                               nnz,
                               dRow,
                               dCol,
                               dVal,
                               CUSPARSE_INDEX_32I,
                               CUSPARSE_INDEX_32I,
                               CUSPARSE_INDEX_BASE_ZERO,
                               CUDA_R_32F);
        std::printf("[CSR malformed create status] %d (success is possible; must validate inputs ourselves)\n",
                    static_cast<int>(st));
        if(st == CUSPARSE_STATUS_SUCCESS)
            cusparseDestroySpMat(A);
        cudaFree(dRow);
        cudaFree(dCol);
        cudaFree(dVal);
    }

    // ---------------- COO descriptor consistency ----------------
    {
        constexpr int m = 4, n = 5, nnz = 8;
        int   hRow[nnz] = {0, 0, 1, 1, 2, 2, 2, 3};
        int   hCol[nnz] = {0, 3, 1, 4, 0, 2, 4, 3};
        float hVal[nnz] = {2.f, -1.f, 3.f, 0.5f, -2.f, 1.25f, 4.f, -0.75f};

        print_status("COO input rule check", check_coo_rules(m, n, nnz, hRow, hCol, 0));

        int *  dRow = nullptr, *dCol = nullptr;
        float* dVal = nullptr;
        cudaMalloc(&dRow, sizeof(hRow));
        cudaMalloc(&dCol, sizeof(hCol));
        cudaMalloc(&dVal, sizeof(hVal));
        cudaMemcpy(dRow, hRow, sizeof(hRow), cudaMemcpyHostToDevice);
        cudaMemcpy(dCol, hCol, sizeof(hCol), cudaMemcpyHostToDevice);
        cudaMemcpy(dVal, hVal, sizeof(hVal), cudaMemcpyHostToDevice);

        cusparseSpMatDescr_t A = nullptr;
        auto                 st =
            cusparseCreateCoo(&A, m, n, nnz, dRow, dCol, dVal, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        print_status("COO create status", st == CUSPARSE_STATUS_SUCCESS);

        cusparseFormat_t fmt{};
        st = cusparseSpMatGetFormat(A, &fmt);
        print_status("COO get format", st == CUSPARSE_STATUS_SUCCESS && fmt == CUSPARSE_FORMAT_COO);

        cusparseIndexBase_t base{};
        st = cusparseSpMatGetIndexBase(A, &base);
        print_status("COO get base", st == CUSPARSE_STATUS_SUCCESS && base == CUSPARSE_INDEX_BASE_ZERO);

        int64_t rows = -1, cols = -1, got_nnz = -1;
        void *  pRow = nullptr, *pCol = nullptr, *pVal = nullptr;
        cusparseIndexType_t it{};
        cudaDataType        dt{};
        st = cusparseCooGet(A, &rows, &cols, &got_nnz, &pRow, &pCol, &pVal, &it, &base, &dt);
        print_status("COO get meta",
                     st == CUSPARSE_STATUS_SUCCESS && rows == m && cols == n && got_nnz == nnz
                         && it == CUSPARSE_INDEX_32I && dt == CUDA_R_32F);
        print_status("COO pointer identity", pRow == dRow && pCol == dCol && pVal == dVal);

        int   row_back[nnz];
        int   col_back[nnz];
        float val_back[nnz];
        cudaMemcpy(row_back, static_cast<int*>(pRow), sizeof(row_back), cudaMemcpyDeviceToHost);
        cudaMemcpy(col_back, static_cast<int*>(pCol), sizeof(col_back), cudaMemcpyDeviceToHost);
        cudaMemcpy(val_back, static_cast<float*>(pVal), sizeof(val_back), cudaMemcpyDeviceToHost);

        bool arrays_ok = std::memcmp(row_back, hRow, sizeof(hRow)) == 0
                         && std::memcmp(col_back, hCol, sizeof(hCol)) == 0;
        for(int i = 0; i < nnz && arrays_ok; ++i)
            arrays_ok = nearf(val_back[i], hVal[i]);
        print_status("COO array roundtrip", arrays_ok);
        print_status("COO roundtrip rule check", check_coo_rules(m, n, nnz, row_back, col_back, 0));

        cusparseDestroySpMat(A);
        cudaFree(dRow);
        cudaFree(dCol);
        cudaFree(dVal);
    }

    cusparseDestroy(h);
    cudaStreamDestroy(stream);
    return 0;
}
