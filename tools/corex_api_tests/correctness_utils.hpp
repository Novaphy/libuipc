#pragma once
/**
 * CPU references and comparisons for corex_api_tests (not production code).
 */
#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

inline void case_begin(const char* symbol)
{
    std::printf("[CASE %s]\n", symbol);
}

inline bool nearly_equal_d(double a, double b, double eps)
{
    return std::abs(a - b) <= eps * (1.0 + std::max(std::abs(a), std::abs(b)));
}

inline bool nearly_equal_f(float a, float b, float eps)
{
    return std::abs(a - b) <= eps * (1.f + std::max(std::abs(a), std::abs(b)));
}

/** On mismatch: fail() and return false. */
inline bool expect_near_d(const char* tag, double gpu, double ref, double eps)
{
    if(nearly_equal_d(gpu, ref, eps))
    {
        ok(tag);
        return true;
    }
    char buf[384];
    std::snprintf(buf, sizeof(buf), "%s gpu=%.17e ref=%.17e", tag, static_cast<double>(gpu), static_cast<double>(ref));
    fail(buf);
    return false;
}

inline bool expect_near_f(const char* tag, float gpu, float ref, float eps)
{
    if(nearly_equal_f(gpu, ref, eps))
    {
        ok(tag);
        return true;
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s gpu=%.8g ref=%.8g", tag, gpu, ref);
    fail(buf);
    return false;
}

inline bool expect_buffer_equal(const char* tag, const void* a, const void* b, size_t n)
{
    if(std::memcmp(a, b, n) == 0)
    {
        ok(tag);
        return true;
    }
    fail(tag);
    return false;
}

/** y = alpha * A * x + beta * y, CSR zero-based, dimensions m x n. */
inline void cpu_csr_spmv_f(int m, int n, const int* rowptr, const int* colidx, const float* val, const float* x,
                           float* y, float alpha, float beta)
{
    for(int i = 0; i < m; ++i)
    {
        double acc = 0;
        for(int p = rowptr[i]; p < rowptr[i + 1]; ++p)
        {
            int j = colidx[p];
            if(j >= 0 && j < n)
                acc += static_cast<double>(val[p]) * static_cast<double>(x[j]);
        }
        const double y0 = beta == 0.f ? 0.0 : static_cast<double>(y[i]);
        y[i] = static_cast<float>(alpha * acc + beta * y0);
    }
}

/** COO: duplicate (row,col) allowed; accumulates. */
inline void cpu_coo_spmv_f(int m, int n, int nnz, const int* row, const int* col, const float* val, const float* x,
                           float* y, float alpha, float beta)
{
    for(int i = 0; i < m; ++i)
        y[i] = beta == 0.f ? 0.f : beta * y[i];
    for(int p = 0; p < nnz; ++p)
    {
        int r = row[p], c = col[p];
        if(r >= 0 && r < m && c >= 0 && c < n)
            y[r] += alpha * val[p] * x[c];
    }
}

/** Row-major A[2][2], x solves A x = b. */
inline bool solve_2x2_double(const double A[4], const double b[2], double x[2])
{
    double a00 = A[0], a01 = A[1], a10 = A[2], a11 = A[3];
    double det = a00 * a11 - a01 * a10;
    if(std::abs(det) < 1e-30)
        return false;
    x[0] = (a11 * b[0] - a01 * b[1]) / det;
    x[1] = (a00 * b[1] - a10 * b[0]) / det;
    return true;
}

/** Row-major A (n×n), Gauss–Jordan on augmented [A|b], n ≤ 32. */
inline bool cpu_dense_solve_rowmajor(int n, const double* A_row, const double* b, double* x)
{
    if(n <= 0 || n > 32)
        return false;
    double A[32][33];
    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < n; ++j)
            A[i][j] = A_row[i * n + j];
        A[i][n] = b[i];
    }
    for(int k = 0; k < n; ++k)
    {
        int    piv = k;
        double maxv = std::abs(A[k][k]);
        for(int i = k + 1; i < n; ++i)
        {
            double v = std::abs(A[i][k]);
            if(v > maxv)
            {
                maxv = v;
                piv  = i;
            }
        }
        if(maxv < 1e-18)
            return false;
        if(piv != k)
        {
            for(int j = 0; j <= n; ++j)
                std::swap(A[k][j], A[piv][j]);
        }
        double akk = A[k][k];
        for(int j = 0; j <= n; ++j)
            A[k][j] /= akk;
        for(int i = 0; i < n; ++i)
        {
            if(i == k)
                continue;
            double f = A[i][k];
            for(int j = 0; j <= n; ++j)
                A[i][j] -= f * A[k][j];
        }
    }
    for(int i = 0; i < n; ++i)
        x[i] = A[i][n];
    return true;
}

inline void cpu_matvec_rowmajor_d(int n, const double* A, const double* v, double* out)
{
    for(int i = 0; i < n; ++i)
    {
        double s = 0;
        for(int j = 0; j < n; ++j)
            s += A[i * n + j] * v[j];
        out[i] = s;
    }
}
