#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#pragma once
#if __has_include(<span>)
#include <span>
namespace uipc
{
/**
 * @brief just an alias for std::span
 */
using std::span;
}  // namespace uipc
#elif __has_include(<cuda/std/span>)
#include <cuda/std/span>
namespace uipc
{
/**
 * @brief fallback alias to cuda::std::span for old host stdlibs
 */
template <typename T, size_t Extent = cuda::std::dynamic_extent>
using span = cuda::std::span<T, Extent>;
}  // namespace uipc
#elif __has_include(<experimental/span>)
#include <experimental/span>
namespace uipc
{
/**
 * @brief fallback alias for older libstdc++
 */
template <typename T, std::size_t Extent = std::experimental::dynamic_extent>
using span = std::experimental::span<T, Extent>;
}  // namespace uipc
#else
#error "Neither <span> nor <experimental/span> is available on this toolchain."
#endif
#else
#pragma once
#include <span>
namespace uipc
{
/**
 * @brief just an alias for std::span
 */
using std::span;
}  // namespace uipc
#endif
