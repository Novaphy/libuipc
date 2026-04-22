#include <muda/buffer/buffer_launch.h>
#include <muda/launch/memory.h>

namespace muda
{
template <typename T>
void DeviceVar<T>::ensure_allocated() const
{
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    if(!m_data)
    {
        // Default-async on Corex 4.4+ (UIPC_COREX_ASYNC_MEMORY=1) so first use
        // does not block on a synchronous cudaMalloc; falls back to the legacy
        // sync path automatically when DEFAULT_ASYNC_ALLOC_FREE is false.
        Memory().alloc(const_cast<T**>(&m_data), sizeof(T));
    }
#endif
}

template <typename T>
DeviceVar<T>::DeviceVar()
{
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    // CoreX frequently stalls when many DeviceVars allocate during SimSystem
    // construction. Defer the cudaMalloc until first use on that platform.
    m_data = nullptr;
#else
    Memory().alloc(&m_data, sizeof(T)).wait();
#endif
}
template <typename T>
DeviceVar<T>::DeviceVar(const T& value)
{
    m_data = nullptr;
    ensure_allocated();
    view().copy_from(&value);
};

template <typename T>
DeviceVar<T>::DeviceVar(const DeviceVar& other)
{
    m_data = nullptr;
    ensure_allocated();
    view().copy_from(other.view());
}

template <typename T>
DeviceVar<T>& DeviceVar<T>::operator=(const DeviceVar<T>& other)
{
    if(this == &other)
        return *this;
    ensure_allocated();
    view().copy_from(other.view());
    return *this;
}

template <typename T>
DeviceVar<T>& DeviceVar<T>::operator=(DeviceVar<T>&& other)
{
    if(this == &other)
        return *this;

    if(m_data)
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        // Default-async on Corex 4.4+; sync on legacy Corex 3.x.
        Memory().free(m_data);
#else
        Memory().free(m_data).wait();
#endif

    m_data = other.m_data;

    other.m_data = nullptr;

    return *this;
}

template <typename T>
DeviceVar<T>& DeviceVar<T>::operator=(CVarView<T> other)
{
    ensure_allocated();
    view().copy_from(other);
    return *this;
}

template <typename T>
void DeviceVar<T>::copy_from(CVarView<T> other)
{
    ensure_allocated();
    view().copy_from(other);
}

template <typename T>
DeviceVar<T>& DeviceVar<T>::operator=(const T& val)
{
    ensure_allocated();
    view().copy_from(&val);
    return *this;
}

template <typename T>
DeviceVar<T>::DeviceVar(DeviceVar&& other) MUDA_NOEXCEPT : m_data(other.m_data)
{
    other.m_data = nullptr;
}

template <typename T>
DeviceVar<T>::operator T() const
{
    T var;
    view().copy_to(&var);
    return var;
}

template <typename T>
Dense<T> DeviceVar<T>::viewer() MUDA_NOEXCEPT
{
    ensure_allocated();
    return Dense<T>(m_data);
}

template <typename T>
CDense<T> DeviceVar<T>::cviewer() const MUDA_NOEXCEPT
{
    ensure_allocated();
    return CDense<T>(m_data);
}

template <typename T>
DeviceVar<T>::~DeviceVar()
{
    if(m_data)
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        // Default-async on Corex 4.4+; sync on legacy Corex 3.x.
        Memory().free(m_data);
#else
        Memory().free(m_data).wait();
#endif
}
}  // namespace muda
