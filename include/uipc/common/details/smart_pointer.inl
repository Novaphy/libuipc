#include <uipc/common/log.h>
namespace uipc
{
template <typename T>
void PmrDeleter<T>::operator()(T* ptr) const
{
    auto      resource = std::pmr::get_default_resource();
    Allocator alloc{resource};
    std::allocator_traits<Allocator>::destroy(alloc, ptr);
    std::allocator_traits<Allocator>::deallocate(alloc, ptr, 1);
}

template <typename T, typename... Args>
U<T> make_unique(Args&&... args)
{
    std::pmr::polymorphic_allocator<T> alloc;
    T*                               ptr = std::allocator_traits<decltype(alloc)>::allocate(alloc, 1);
    std::allocator_traits<decltype(alloc)>::construct(alloc, ptr, std::forward<Args>(args)...);
    return U<T>(ptr, PmrDeleter<T>{});
}

template <typename DstT, typename SrcT>
U<DstT> static_pointer_cast(U<SrcT>&& src)
{
    return U<DstT>(src.release());
}

template <typename T, typename... Args>
S<T> make_shared(Args&&... args)
{
    auto resource = std::pmr::get_default_resource();
    std::pmr::polymorphic_allocator<T> alloc{resource};
    T* ptr = std::allocator_traits<decltype(alloc)>::allocate(alloc, 1);
    std::allocator_traits<decltype(alloc)>::construct(alloc, ptr, std::forward<Args>(args)...);
    return std::shared_ptr<T>(ptr, PmrDeleter<T>{});
}
}  // namespace uipc