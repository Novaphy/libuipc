/**
 * corex_runtime_extended — A-class cuda_runtime symbols from api_test_matrix.md (一符号一测).
 *
 * See file header comment block in api_test_matrix.md § A Runtime.
 */
#include "correctness_utils.hpp"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace
{
std::atomic<int> g_host_cb_count{0};

void CUDART_CB streamCallback(cudaStream_t, cudaError_t, void*)
{
    ++g_host_cb_count;
}

__global__ void k_inc(float* p, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        p[i] += 1.f;
}

void CUDART_CB hostFunc(void* userData)
{
    int* p = static_cast<int*>(userData);
    ++(*p);
}

}  // namespace

int main()
{
    std::printf("=== corex_runtime_extended ===\n");

    int nDev = 0;
    {
        case_begin("cudaGetDeviceCount");
        cuda_ok(cudaGetDeviceCount(&nDev), "cudaGetDeviceCount");
    }
    if(nDev < 1)
    {
        fail("no device");
        return 1;
    }

    cudaDeviceProp prop{};
    {
        case_begin("cudaGetDeviceProperties");
        cuda_ok(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    }
    int dev = -1;
    {
        case_begin("cudaGetDevice");
        cuda_ok(cudaGetDevice(&dev), "cudaGetDevice");
    }
    {
        case_begin("cudaSetDevice");
        cuda_ok(cudaSetDevice(0), "cudaSetDevice");
    }

    {
        case_begin("cudaGetLastError");
        cudaGetLastError();
        cuda_ok(cudaGetLastError(), "cudaGetLastError after clear");
    }
    {
        case_begin("cudaGetErrorName");
        const char* en = cudaGetErrorName(cudaSuccess);
        if(!en)
            fail("cudaGetErrorName");
        else
            ok("cudaGetErrorName");
    }
    {
        case_begin("cudaGetErrorString");
        const char* es = cudaGetErrorString(cudaSuccess);
        if(!es)
            fail("cudaGetErrorString");
        else
            ok("cudaGetErrorString");
    }

    cudaStream_t stream{};
    {
        case_begin("cudaStreamCreateWithFlags");
        cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    }

#if CUDART_VERSION >= 11020
    {
        case_begin("cudaMallocAsync");
        case_begin("cudaFreeAsync");
        float*        d_async = nullptr;
        cudaError_t   eMallocA = cudaMallocAsync(&d_async, 64, stream);
        if(eMallocA == cudaSuccess)
        {
            cuda_ok(cudaFreeAsync(d_async, stream), "cudaFreeAsync");
            ok("cudaMallocAsync / cudaFreeAsync");
        }
        else
        {
            std::fprintf(stderr, "[WARN] cudaMallocAsync: %s\n", cudaGetErrorString(eMallocA));
            ++g_warnings();
        }
    }
#else
    warn("cudaMallocAsync/cudaFreeAsync skipped (CUDART_VERSION < 11020)");
#endif

    constexpr int kN1d = 512;
    constexpr int k2d  = 16;
    float           h1[kN1d];
    for(int i = 0; i < kN1d; ++i)
        h1[i] = static_cast<float>(i);

    float* d1 = nullptr;
    {
        case_begin("cudaMalloc");
        cuda_ok(cudaMalloc(&d1, static_cast<size_t>(kN1d) * sizeof(float)), "cudaMalloc");
    }
    {
        case_begin("cudaMemcpyAsync");
        cuda_ok(cudaMemcpyAsync(d1, h1, sizeof(h1), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
    }
    {
        case_begin("cudaMemsetAsync");
        cuda_ok(cudaMemsetAsync(d1, 0, sizeof(float) * 128, stream), "cudaMemsetAsync");
    }
    {
        case_begin("cudaStreamSynchronize");
        cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
    {
        float hrd[kN1d];
        cudaMemcpy(hrd, d1, sizeof(hrd), cudaMemcpyDeviceToHost);
        float hex[kN1d];
        std::memcpy(hex, h1, sizeof(h1));
        for(int i = 0; i < 128; ++i)
            hex[i] = 0.f;
        expect_buffer_equal("cudaMemcpyAsync+MemsetAsync readback", hrd, hex, sizeof(hex));
    }

    size_t       pitch = 0;
    float*       d2 = nullptr;
    {
        case_begin("cudaMallocPitch");
        cuda_ok(cudaMallocPitch(&d2, &pitch, static_cast<size_t>(k2d) * sizeof(float), static_cast<size_t>(k2d)),
                "cudaMallocPitch");
    }
    {
        case_begin("cudaMemcpy2DAsync");
        cuda_ok(cudaMemcpy2DAsync(d2, pitch, h1, static_cast<size_t>(k2d) * sizeof(float),
                                  static_cast<size_t>(k2d) * sizeof(float), static_cast<size_t>(k2d),
                                  cudaMemcpyHostToDevice, stream),
            "cudaMemcpy2DAsync");
    }
    {
        case_begin("cudaMemset2DAsync");
        cuda_ok(cudaMemset2DAsync(d2, pitch, 0, static_cast<size_t>(k2d) * sizeof(float), 8, stream),
                "cudaMemset2DAsync");
    }
    cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize after 2D");
    {
        float z[k2d];
        for(int c = 0; c < k2d; ++c)
            z[c] = 0.f;
        for(int r = 0; r < 8; ++r)
        {
            float row[k2d];
            cudaMemcpy2D(row, sizeof(float) * static_cast<size_t>(k2d), reinterpret_cast<char*>(d2) + r * pitch, pitch,
                         static_cast<size_t>(k2d) * sizeof(float), 1, cudaMemcpyDeviceToHost);
            char tag[64];
            std::snprintf(tag, sizeof(tag), "cudaMemset2DAsync row%d zero", r);
            expect_buffer_equal(tag, row, z, sizeof(z));
        }
        float rlast[k2d];
        cudaMemcpy2D(rlast, sizeof(float) * static_cast<size_t>(k2d), reinterpret_cast<char*>(d2) + 15 * pitch, pitch,
                     static_cast<size_t>(k2d) * sizeof(float), 1, cudaMemcpyDeviceToHost);
        float r15e[k2d];
        for(int c = 0; c < k2d; ++c)
            r15e[c] = h1[15 * k2d + c];
        expect_buffer_equal("cudaMemcpy2DAsync row15 intact", rlast, r15e, sizeof(r15e));
    }

    cudaExtent     ext = make_cudaExtent(8 * sizeof(float), 4, 4);
    cudaPitchedPtr pp{};
    {
        case_begin("cudaMalloc3D");
        cuda_ok(cudaMalloc3D(&pp, ext), "cudaMalloc3D");
    }
    {
        case_begin("cudaMemcpy3DAsync");
        cudaMemcpy3DParms p3{};
        std::memset(&p3, 0, sizeof(p3));
        p3.srcPtr = make_cudaPitchedPtr(h1, 8 * sizeof(float), 8, 8);
        p3.dstPtr = pp;
        p3.extent = make_cudaExtent(8 * sizeof(float), 4, 4);
        p3.kind   = cudaMemcpyHostToDevice;
        cuda_ok(cudaMemcpy3DAsync(&p3, stream), "cudaMemcpy3DAsync");
    }
    {
        case_begin("cudaMemset3DAsync");
        cuda_ok(cudaMemset3DAsync(pp, 0, make_cudaExtent(4 * sizeof(float), 2, 2), stream), "cudaMemset3DAsync");
    }
    cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize after 3D");
    {
        float sample[1];
        cudaMemcpy(sample, pp.ptr, sizeof(float), cudaMemcpyDeviceToHost);
        float zero = 0.f;
        expect_buffer_equal("cudaMemset3DAsync corner", sample, &zero, sizeof(float));
    }

    cudaFree(d1);
    cudaFree(d2);
    cudaFree(pp.ptr);

    int h2 = 7, h2o = -1;
    int* d3 = nullptr;
    {
        case_begin("cudaMalloc");
        cuda_ok(cudaMalloc(&d3, sizeof(int)), "cudaMalloc int");
    }
    {
        case_begin("cudaMemcpy");
        cuda_ok(cudaMemcpy(d3, &h2, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");
    }
    {
        case_begin("cudaMemset");
        cuda_ok(cudaMemset(d3, 0, sizeof(int)), "cudaMemset");
    }
    cuda_ok(cudaMemcpy(&h2o, d3, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    {
        int z = 0;
        expect_buffer_equal("cudaMemset int", &h2o, &z, sizeof(int));
    }
    cudaFree(d3);

    cudaEvent_t ev0{}, ev1{};
    {
        case_begin("cudaEventCreateWithFlags");
        cuda_ok(cudaEventCreateWithFlags(&ev0, cudaEventDefault), "cudaEventCreateWithFlags ev0");
    }
    {
        case_begin("cudaEventCreateWithFlags");
        cuda_ok(cudaEventCreateWithFlags(&ev1, cudaEventDefault), "cudaEventCreateWithFlags ev1");
    }
    {
        case_begin("cudaEventRecord");
        cuda_ok(cudaEventRecord(ev0, stream), "cudaEventRecord");
    }
#if CUDART_VERSION >= 11000
    {
        case_begin("cudaEventRecordWithFlags");
        cuda_ok(cudaEventRecordWithFlags(ev1, stream, 0), "cudaEventRecordWithFlags");
    }
#else
    cuda_ok(cudaEventRecord(ev1, stream), "cudaEventRecord (stand-in for RecordWithFlags)");
    warn("cudaEventRecordWithFlags skipped (CUDART_VERSION < 11000)");
#endif
    {
        case_begin("cudaStreamWaitEvent");
        cuda_ok(cudaStreamWaitEvent(stream, ev0, 0), "cudaStreamWaitEvent");
        cuda_ok(cudaStreamWaitEvent(stream, ev1, 0), "cudaStreamWaitEvent ev1");
    }
    {
        case_begin("cudaEventQuery");
        cudaError_t q = cudaEventQuery(ev0);
        if(q != cudaSuccess && q != cudaErrorNotReady)
            cuda_ok(q, "cudaEventQuery");
        else
            ok("cudaEventQuery");
    }
    {
        case_begin("cudaEventSynchronize");
        cuda_ok(cudaEventSynchronize(ev0), "cudaEventSynchronize");
    }
    {
        case_begin("cudaEventElapsedTime");
        float ms = -1.f;
        cuda_ok(cudaEventElapsedTime(&ms, ev0, ev1), "cudaEventElapsedTime");
        if(!(ms >= 0.f) || std::isnan(ms))
            fail("cudaEventElapsedTime range");
        else
            ok("cudaEventElapsedTime sane");
    }
    {
        case_begin("cudaEventDestroy");
        cuda_ok(cudaEventDestroy(ev0), "cudaEventDestroy ev0");
    }
    {
        case_begin("cudaEventDestroy");
        cuda_ok(cudaEventDestroy(ev1), "cudaEventDestroy ev1");
    }

    g_host_cb_count = 0;
    {
        case_begin("cudaStreamAddCallback");
        cuda_ok(cudaStreamAddCallback(stream, streamCallback, nullptr, 0), "cudaStreamAddCallback");
    }
    {
        case_begin("cudaLaunchHostFunc");
        int hf_cnt = 0;
        cuda_ok(cudaLaunchHostFunc(stream, hostFunc, &hf_cnt), "cudaLaunchHostFunc");
        cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize hostfunc");
        if(g_host_cb_count.load() < 1)
            warn("stream callback count unexpected");
        if(hf_cnt != 1)
            warn("cudaLaunchHostFunc count unexpected");
        else
            ok("cudaLaunchHostFunc effect");
    }

    {
        case_begin("cudaOccupancyMaxPotentialBlockSize");
        int minGrid = 0, blockSize = 0;
        cuda_ok(cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, k_inc, 0, 0),
                "cudaOccupancyMaxPotentialBlockSize");
        ok("cudaOccupancyMaxPotentialBlockSize");
    }

    {
        case_begin("cudaProfilerStart");
        cudaError_t pe = cudaProfilerStart();
        if(pe != cudaSuccess)
        {
            std::fprintf(stderr, "[WARN] cudaProfilerStart: %s\n", cudaGetErrorString(pe));
            ++g_warnings();
        }
        else
        {
            case_begin("cudaProfilerStop");
            cudaProfilerStop();
            ok("cudaProfilerStart/Stop");
        }
    }

    {
        case_begin("cudaDeviceSynchronize");
        cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    {
        case_begin("cudaStreamDestroy");
        cuda_ok(cudaStreamDestroy(stream), "cudaStreamDestroy");
    }

    print_summary("corex_runtime_extended");
    return g_failures() > 0 ? 1 : 0;
}
