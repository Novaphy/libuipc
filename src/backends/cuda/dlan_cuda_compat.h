#pragma once

#if defined(UIPC_DLAN_COMPAT) && UIPC_DLAN_COMPAT
#include <cuda_runtime_api.h>

// The C++11 device-attribute cache in DLAN's bundled CUB can retain an
// invalid device result while the backend DSO is being initialized.  Selecting
// CUB's uncached compatibility branch affects only these host-side attribute
// helpers; libuipc and all kernels are still compiled as C++20.
#ifndef CUB_CPP_DIALECT
#define CUB_CPP_DIALECT 2003
#endif

// DLAN's CUDA-11 CUB headers make CUB_RUNTIME_FUNCTION host-only whenever
// relocatable device code is disabled.  libuipc uses this macro to annotate
// reduction and selection functors, which must remain callable by kernels.
#ifndef CUB_RUNTIME_FUNCTION
#define CUB_RUNTIME_FUNCTION __host__ __device__
#endif
#ifndef CUB_RUNTIME_ENABLED
#define CUB_RUNTIME_ENABLED
#endif

// DLAN returns cudaErrorInvalidDevice when old CUB asks for the attributes of
// its synthetic EmptyKernel.  CUB only performs this query to select an
// architecture policy; the backend is compiled for dlgput64/SM 7.0 already.
// Preserve the real runtime query when it works and provide only that policy
// metadata for this known failure case.
template <typename Function>
inline cudaError_t uipc_dlan_cuda_func_get_attributes(cudaFuncAttributes* attr, Function func)
{
    const cudaError_t result =
        ::cudaFuncGetAttributes(attr, reinterpret_cast<const void*>(func));
    if(result != cudaErrorInvalidDevice && result != cudaErrorNotSupported)
        return result;

    // Clear the runtime's last-error slot before CUB launches another kernel.
    (void)::cudaGetLastError();
    *attr               = cudaFuncAttributes{};
    attr->binaryVersion = 70;
    attr->ptxVersion    = 70;
    return cudaSuccess;
}

#define cudaFuncGetAttributes uipc_dlan_cuda_func_get_attributes
#endif
