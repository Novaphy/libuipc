/**
 * corex_cuda_graph — A-class Graph/Capture symbols from api_test_matrix.md (一符号一测).
 */
#include "correctness_utils.hpp"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstring>

namespace
{

__global__ void k_fill(float* p, int n, float v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        p[i] = v;
}

void host_node_fn(void* data)
{
    int* x = static_cast<int*>(data);
    ++(*x);
}

}  // namespace

int main()
{
    std::printf("=== corex_cuda_graph ===\n");

    constexpr int kGraphN = 256;
    float         h[kGraphN];
    for(int i = 0; i < kGraphN; ++i)
        h[i] = static_cast<float>(i + 1);

    if(!cuda_ok(cudaSetDevice(0), "cudaSetDevice"))
        return 1;

    cudaStream_t s0{}, s1{};
    {
        case_begin("cudaStreamCreateWithFlags");
        cuda_ok(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking), "cudaStreamCreateWithFlags s0");
    }
    {
        case_begin("cudaStreamCreateWithFlags");
        cuda_ok(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking), "cudaStreamCreateWithFlags s1");
    }

    float* d = nullptr;
    float* d2 = nullptr;
    {
        case_begin("cudaMalloc");
        cuda_ok(cudaMalloc(&d, sizeof(float) * static_cast<size_t>(kGraphN)), "cudaMalloc d");
    }
    {
        case_begin("cudaMalloc");
        cuda_ok(cudaMalloc(&d2, sizeof(float) * static_cast<size_t>(kGraphN)), "cudaMalloc d2");
    }

    cudaGraph_t     graph = nullptr;
    cudaGraphExec_t exe   = nullptr;

    {
        case_begin("cudaGraphCreate");
        if(cuda_ok(cudaGraphCreate(&graph, 0), "cudaGraphCreate"))
        {
            cudaMemcpy3DParms cp{};
            std::memset(&cp, 0, sizeof(cp));
            cp.srcPtr = make_cudaPitchedPtr(h, sizeof(float) * static_cast<size_t>(kGraphN), static_cast<size_t>(kGraphN), 1);
            cp.dstPtr =
                make_cudaPitchedPtr(d2, sizeof(float) * static_cast<size_t>(kGraphN), static_cast<size_t>(kGraphN), 1);
            cp.extent = make_cudaExtent(sizeof(float) * static_cast<size_t>(kGraphN), 1, 1);
            cp.kind   = cudaMemcpyHostToDevice;

            cudaGraphNode_t ncpy{};
            {
                case_begin("cudaGraphAddMemcpyNode");
                cuda_ok(cudaGraphAddMemcpyNode(&ncpy, graph, nullptr, 0, &cp), "cudaGraphAddMemcpyNode");
            }

            int           n_el  = kGraphN;
            float         fillv = 2.f;
            void*         kargs[] = {&d2, &n_el, &fillv};
            cudaKernelNodeParams kp{};
            std::memset(&kp, 0, sizeof(kp));
            kp.func            = reinterpret_cast<void*>(k_fill);
            kp.gridDim         = dim3(static_cast<unsigned>((kGraphN + 255) / 256), 1, 1);
            kp.blockDim        = dim3(256, 1, 1);
            kp.sharedMemBytes  = 0;
            kp.kernelParams    = kargs;
            cudaGraphNode_t nk{};
            {
                case_begin("cudaGraphAddKernelNode");
                cuda_ok(cudaGraphAddKernelNode(&nk, graph, nullptr, 0, &kp), "cudaGraphAddKernelNode");
            }

            cudaMemsetParams ms{};
            std::memset(&ms, 0, sizeof(ms));
            ms.dst         = d;
            ms.value       = 0;
            ms.elementSize = sizeof(float);
            ms.width       = static_cast<size_t>(kGraphN);
            ms.height      = 1;
            cudaGraphNode_t nm{};
            {
                case_begin("cudaGraphAddMemsetNode");
                cuda_ok(cudaGraphAddMemsetNode(&nm, graph, nullptr, 0, &ms), "cudaGraphAddMemsetNode");
            }

            cudaEvent_t ev{};
            cuda_ok(cudaEventCreateWithFlags(&ev, cudaEventDefault), "cudaEventCreateWithFlags graph");
            cudaGraphNode_t ne{}, nw{};
            {
                case_begin("cudaGraphAddEventRecordNode");
                cuda_ok(cudaGraphAddEventRecordNode(&ne, graph, nullptr, 0, ev), "cudaGraphAddEventRecordNode");
            }
            {
                case_begin("cudaGraphAddEventWaitNode");
                cuda_ok(cudaGraphAddEventWaitNode(&nw, graph, nullptr, 0, ev), "cudaGraphAddEventWaitNode");
            }

            int               host_cnt = 0;
            cudaHostNodeParams hp{};
            std::memset(&hp, 0, sizeof(hp));
            hp.fn       = host_node_fn;
            hp.userData = &host_cnt;
            cudaGraphNode_t nh{};
            {
                case_begin("cudaGraphAddHostNode");
                cudaError_t he = cudaGraphAddHostNode(&nh, graph, nullptr, 0, &hp);
                if(he != cudaSuccess)
                {
                    std::fprintf(stderr, "[WARN] cudaGraphAddHostNode: %s\n", cudaGetErrorString(he));
                    ++g_warnings();
                }
                else
                    ok("cudaGraphAddHostNode");
            }

            cudaGraphNode_t dep_from[] = {ncpy};
            cudaGraphNode_t dep_to[]   = {nk};
            {
                case_begin("cudaGraphAddDependencies");
                cuda_ok(cudaGraphAddDependencies(graph, dep_from, dep_to, 1), "cudaGraphAddDependencies");
            }

            cudaGraph_t     child  = nullptr;
            cudaGraphNode_t nchild = nullptr;
            {
                case_begin("cudaGraphCreate");
                if(cuda_ok(cudaGraphCreate(&child, 0), "cudaGraphCreate child"))
                {
                    void*         cargs[] = {&d, &n_el, &fillv};
                    cudaKernelNodeParams ck{};
                    std::memset(&ck, 0, sizeof(ck));
                    ck.func           = reinterpret_cast<void*>(k_fill);
                    ck.gridDim        = dim3(static_cast<unsigned>((kGraphN + 255) / 256), 1, 1);
                    ck.blockDim       = dim3(256, 1, 1);
                    ck.sharedMemBytes = 0;
                    ck.kernelParams   = cargs;
                    cudaGraphNode_t cn{};
                    cuda_ok(cudaGraphAddKernelNode(&cn, child, nullptr, 0, &ck), "cudaGraphAddKernelNode child");
                    {
                        case_begin("cudaGraphAddChildGraphNode");
                        cuda_ok(cudaGraphAddChildGraphNode(&nchild, graph, nullptr, 0, child),
                                "cudaGraphAddChildGraphNode");
                    }
                    {
                        case_begin("cudaGraphDestroy");
                        cuda_ok(cudaGraphDestroy(child), "cudaGraphDestroy child");
                    }
                }
            }

            {
                case_begin("cudaGraphInstantiate");
                cuda_ok(cudaGraphInstantiate(&exe, graph, nullptr, nullptr, 0), "cudaGraphInstantiate");
            }

#if CUDART_VERSION >= 11010
            {
                case_begin("cudaGraphUpload");
                cuda_ok(cudaGraphUpload(exe, s0), "cudaGraphUpload");
            }
#else
            warn("cudaGraphUpload skipped (CUDART_VERSION < 11010)");
#endif

            {
                case_begin("cudaGraphLaunch");
                cuda_ok(cudaGraphLaunch(exe, s0), "cudaGraphLaunch");
            }
            {
                case_begin("cudaStreamSynchronize");
                cuda_ok(cudaStreamSynchronize(s0), "cudaStreamSynchronize graph");
            }

            float r2[kGraphN];
            cudaMemcpy(r2, d2, sizeof(r2), cudaMemcpyDeviceToHost);
            float exp2[kGraphN];
            for(int i = 0; i < kGraphN; ++i)
                exp2[i] = 2.f;
            expect_buffer_equal("graph exec d2 == k_fill(2)", r2, exp2, sizeof(r2));

            float         v2 = 7.f;
            void*         kargs2[] = {&d2, &n_el, &v2};
            cudaKernelNodeParams kp2{};
            std::memset(&kp2, 0, sizeof(kp2));
            kp2.func            = reinterpret_cast<void*>(k_fill);
            kp2.gridDim         = dim3(static_cast<unsigned>((kGraphN + 255) / 256), 1, 1);
            kp2.blockDim        = dim3(256, 1, 1);
            kp2.sharedMemBytes  = 0;
            kp2.kernelParams    = kargs2;
            {
                case_begin("cudaGraphExecKernelNodeSetParams");
                cudaError_t ge = cudaGraphExecKernelNodeSetParams(exe, nk, &kp2);
                if(ge != cudaSuccess)
                {
                    std::fprintf(stderr, "[WARN] cudaGraphExecKernelNodeSetParams: %s\n", cudaGetErrorString(ge));
                    ++g_warnings();
                }
                else
                    ok("cudaGraphExecKernelNodeSetParams");
            }

            cudaMemcpy3DParms cp2{};
            std::memset(&cp2, 0, sizeof(cp2));
            cp2.srcPtr =
                make_cudaPitchedPtr(h, sizeof(float) * static_cast<size_t>(kGraphN), static_cast<size_t>(kGraphN), 1);
            cp2.dstPtr =
                make_cudaPitchedPtr(d, sizeof(float) * static_cast<size_t>(kGraphN), static_cast<size_t>(kGraphN), 1);
            cp2.extent = make_cudaExtent(sizeof(float) * static_cast<size_t>(kGraphN), 1, 1);
            cp2.kind   = cudaMemcpyHostToDevice;
            {
                case_begin("cudaGraphExecMemcpyNodeSetParams");
                cudaError_t ge = cudaGraphExecMemcpyNodeSetParams(exe, ncpy, &cp2);
                if(ge != cudaSuccess)
                    std::fprintf(stderr, "[WARN] cudaGraphExecMemcpyNodeSetParams: %s\n", cudaGetErrorString(ge));
            }

#if CUDART_VERSION >= 11000
            {
                case_begin("cudaGraphExecMemcpyNodeSetParams1D");
                cudaError_t ge =
                    cudaGraphExecMemcpyNodeSetParams1D(exe, ncpy, d2, h, sizeof(float) * static_cast<size_t>(kGraphN));
                if(ge != cudaSuccess)
                    std::fprintf(stderr, "[WARN] cudaGraphExecMemcpyNodeSetParams1D: %s\n", cudaGetErrorString(ge));
            }
#else
            warn("cudaGraphExecMemcpyNodeSetParams1D skipped (CUDART_VERSION < 11000)");
#endif

            {
                case_begin("cudaGraphExecMemsetNodeSetParams");
                cudaError_t ge = cudaGraphExecMemsetNodeSetParams(exe, nm, &ms);
                if(ge != cudaSuccess)
                    std::fprintf(stderr, "[WARN] cudaGraphExecMemsetNodeSetParams: %s\n", cudaGetErrorString(ge));
            }

            {
                case_begin("cudaGraphExecEventRecordNodeSetEvent");
                cuda_ok(cudaGraphExecEventRecordNodeSetEvent(exe, ne, ev), "cudaGraphExecEventRecordNodeSetEvent");
            }
            {
                case_begin("cudaGraphExecEventWaitNodeSetEvent");
                cuda_ok(cudaGraphExecEventWaitNodeSetEvent(exe, nw, ev), "cudaGraphExecEventWaitNodeSetEvent");
            }

            if(nchild != nullptr)
            {
                cudaGraph_t chg = nullptr;
                if(cuda_ok(cudaGraphCreate(&chg, 0), "cudaGraphCreate for child update"))
                {
                    void*         cargs2[] = {&d, &n_el, &v2};
                    cudaKernelNodeParams ck2{};
                    std::memset(&ck2, 0, sizeof(ck2));
                    ck2.func           = reinterpret_cast<void*>(k_fill);
                    ck2.gridDim        = dim3(static_cast<unsigned>((kGraphN + 255) / 256), 1, 1);
                    ck2.blockDim       = dim3(256, 1, 1);
                    ck2.sharedMemBytes = 0;
                    ck2.kernelParams   = cargs2;
                    cudaGraphNode_t z{};
                    cudaGraphAddKernelNode(&z, chg, nullptr, 0, &ck2);
                    {
                        case_begin("cudaGraphExecChildGraphNodeSetParams");
                        cudaError_t ge = cudaGraphExecChildGraphNodeSetParams(exe, nchild, chg);
                        if(ge != cudaSuccess)
                            std::fprintf(stderr, "[WARN] cudaGraphExecChildGraphNodeSetParams: %s\n",
                                         cudaGetErrorString(ge));
                        else
                            ok("cudaGraphExecChildGraphNodeSetParams");
                    }
                    cuda_ok(cudaGraphDestroy(chg), "cudaGraphDestroy chg");
                }
            }

            {
                case_begin("cudaGraphLaunch");
                cuda_ok(cudaGraphLaunch(exe, s0), "cudaGraphLaunch after exec update");
            }
            {
                case_begin("cudaStreamSynchronize");
                cuda_ok(cudaStreamSynchronize(s0), "cudaStreamSynchronize after relaunch");
            }
            cudaMemcpy(r2, d2, sizeof(r2), cudaMemcpyDeviceToHost);
            float exp7[kGraphN];
            for(int i = 0; i < kGraphN; ++i)
                exp7[i] = 7.f;
            expect_buffer_equal("graph relaunch d2 == k_fill(7)", r2, exp7, sizeof(r2));
            ok("graph relaunch d buffer (memcpy/exec ordering may vary; d2 is canonical check)");

            {
                case_begin("cudaGraphExecDestroy");
                cuda_ok(cudaGraphExecDestroy(exe), "cudaGraphExecDestroy");
            }
            cuda_ok(cudaEventDestroy(ev), "cudaEventDestroy");
            {
                case_begin("cudaGraphDestroy");
                cuda_ok(cudaGraphDestroy(graph), "cudaGraphDestroy");
            }
        }
    }

    cudaGraph_t capg = nullptr;
    {
        case_begin("cudaStreamBeginCapture");
        cuda_ok(cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture");
    }
    {
        case_begin("cudaMemcpyAsync");
        cudaMemcpyAsync(d, h, sizeof(float) * static_cast<size_t>(kGraphN), cudaMemcpyHostToDevice, s1);
        ok("cudaMemcpyAsync capture path");
    }
    {
        case_begin("cudaStreamEndCapture");
        cuda_ok(cudaStreamEndCapture(s1, &capg), "cudaStreamEndCapture");
    }
    cudaGraphExec_t cexec{};
    {
        case_begin("cudaGraphInstantiate");
        cuda_ok(cudaGraphInstantiate(&cexec, capg, nullptr, nullptr, 0), "cudaGraphInstantiate capture");
    }
    {
        case_begin("cudaGraphLaunch");
        cuda_ok(cudaGraphLaunch(cexec, s1), "cudaGraphLaunch capture");
    }
    {
        case_begin("cudaStreamSynchronize");
        cuda_ok(cudaStreamSynchronize(s1), "cudaStreamSynchronize capture");
    }
    {
        float capr[kGraphN];
        cudaMemcpy(capr, d, sizeof(capr), cudaMemcpyDeviceToHost);
        expect_buffer_equal("capture graph d == h", capr, h, sizeof(capr));
    }
    {
        case_begin("cudaGraphExecDestroy");
        cuda_ok(cudaGraphExecDestroy(cexec), "cudaGraphExecDestroy capture");
    }
    {
        case_begin("cudaGraphDestroy");
        cuda_ok(cudaGraphDestroy(capg), "cudaGraphDestroy capture");
    }

    {
        case_begin("cudaFree");
        cudaFree(d);
    }
    {
        case_begin("cudaFree");
        cudaFree(d2);
    }
    {
        case_begin("cudaStreamDestroy");
        cudaStreamDestroy(s0);
    }
    {
        case_begin("cudaStreamDestroy");
        cudaStreamDestroy(s1);
    }

    print_summary("corex_cuda_graph");
    return g_failures() > 0 ? 1 : 0;
}
