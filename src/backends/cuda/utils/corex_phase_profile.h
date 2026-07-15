#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace uipc::backend::cuda::corex_profile
{
inline bool enabled()
{
#if defined(UIPC_COREX_ENABLE_PHASE_PROFILE) && UIPC_COREX_ENABLE_PHASE_PROFILE
    static const bool enabled = std::getenv("UIPC_COREX_PHASE_PROFILE") != nullptr;
    return enabled;
#else
    return false;
#endif
}

inline double now_ms()
{
    if(!enabled())
        return 0.0;
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
#if defined(UIPC_COREX_ENABLE_PHASE_PROFILE) && UIPC_COREX_ENABLE_PHASE_PROFILE
        : m_category(category)
        , m_name(name)
        , m_frame(frame)
        , m_newton(newton)
        , m_iter(iter)
        , m_enabled(enabled())
        , m_start(m_enabled ? now_ms() : 0.0)
    {
    }
#else
    {
    }
#endif

    ~ScopedPhase()
    {
#if defined(UIPC_COREX_ENABLE_PHASE_PROFILE) && UIPC_COREX_ENABLE_PHASE_PROFILE
        if(m_enabled)
            log_phase(m_category, m_name, m_frame, m_newton, m_iter, now_ms() - m_start);
#endif
    }

  private:
#if defined(UIPC_COREX_ENABLE_PHASE_PROFILE) && UIPC_COREX_ENABLE_PHASE_PROFILE
    const char* m_category;
    const char* m_name;
    long long   m_frame;
    long long   m_newton;
    long long   m_iter;
    bool        m_enabled;
    double      m_start;
#endif
};
}  // namespace uipc::backend::cuda::corex_profile
