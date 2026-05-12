#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace uipc::backend::cuda::corex_profile
{
inline bool enabled()
{
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    return std::getenv("UIPC_COREX_PHASE_PROFILE") != nullptr;
#else
    return false;
#endif
}

inline double now_ms()
{
    using clock = std::chrono::steady_clock;
    auto now = clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(now).count();
}

inline void log_phase(const char* category,
                      const char* name,
                      long long   frame,
                      long long   newton,
                      long long   iter,
                      double      elapsed_ms)
{
    if(!enabled())
        return;
    std::fprintf(stderr,
                 "[corex_phase] category=%s name=%s frame=%lld newton=%lld iter=%lld elapsed_ms=%.3f\n",
                 category,
                 name,
                 frame,
                 newton,
                 iter,
                 static_cast<float>(elapsed_ms));
}

class ScopedPhase
{
  public:
    ScopedPhase(const char* category,
                const char* name,
                long long   frame  = -1,
                long long   newton = -1,
                long long   iter   = -1)
        : m_category(category)
        , m_name(name)
        , m_frame(frame)
        , m_newton(newton)
        , m_iter(iter)
        , m_start(now_ms())
        , m_enabled(enabled())
    {
    }

    ~ScopedPhase()
    {
        if(m_enabled)
            log_phase(m_category, m_name, m_frame, m_newton, m_iter, now_ms() - m_start);
    }

  private:
    const char* m_category;
    const char* m_name;
    long long   m_frame;
    long long   m_newton;
    long long   m_iter;
    double      m_start;
    bool        m_enabled;
};
}  // namespace uipc::backend::cuda::corex_profile
