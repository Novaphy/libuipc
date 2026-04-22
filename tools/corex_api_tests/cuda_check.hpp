#pragma once
/**
 * Shared helpers for corex_api_tests binaries (same semantics as corex_cuda_api_probe).
 */
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

inline int& g_failures()
{
    static int v = 0;
    return v;
}

inline int& g_warnings()
{
    static int v = 0;
    return v;
}

/** Incremented by each `[OK]` line from ok(); use as coarse pass count vs failures/warnings. */
inline int& g_ok_count()
{
    static int v = 0;
    return v;
}

inline void fail(const char* msg)
{
    std::fprintf(stderr, "[FAIL] %s\n", msg);
    ++g_failures();
}

inline void warn(const char* msg)
{
    std::fprintf(stderr, "[WARN] %s\n", msg);
    ++g_warnings();
}

inline void ok(const char* msg)
{
    std::printf("[OK]   %s\n", msg);
    ++g_ok_count();
}

inline bool cuda_ok(cudaError_t e, const char* what)
{
    if(e != cudaSuccess)
    {
        std::fprintf(stderr, "[FAIL] %s: %s\n", what, cudaGetErrorString(e));
        ++g_failures();
        return false;
    }
    return true;
}

inline void print_summary(const char* name)
{
    std::printf("=== %s: ok=%d failures=%d warnings=%d ===\n", name, g_ok_count(), g_failures(), g_warnings());
}
