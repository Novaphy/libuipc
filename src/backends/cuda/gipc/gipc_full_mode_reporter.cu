#include <sim_system.h>
#include <uipc/common/log.h>

namespace uipc::backend::cuda
{
class GIPCFullModeReporter final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

  protected:
    void do_build() override
    {
#if defined(UIPC_ENABLE_GIPC_FULL) && UIPC_ENABLE_GIPC_FULL
        logger::info(
            "GIPCFullModeReporter: full={}, native_contact={}, matrix_free={}, friction={}, mas={}, rank={}",
            1,
#if defined(UIPC_ENABLE_GIPC_NATIVE_CONTACT) && UIPC_ENABLE_GIPC_NATIVE_CONTACT
            1,
#else
            0,
#endif
#if defined(UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE) && UIPC_ENABLE_GIPC_CONTACT_MATRIX_FREE
            1,
#else
            0,
#endif
#if defined(UIPC_ENABLE_GIPC_FRICTION) && UIPC_ENABLE_GIPC_FRICTION
            1,
#else
            0,
#endif
#if defined(UIPC_ENABLE_GIPC_MAS) && UIPC_ENABLE_GIPC_MAS
            1,
#else
            0,
#endif
#if defined(UIPC_GIPC_RANK)
            UIPC_GIPC_RANK
#else
            1
#endif
        );
#endif
    }
};

REGISTER_SIM_SYSTEM(GIPCFullModeReporter);
}  // namespace uipc::backend::cuda
