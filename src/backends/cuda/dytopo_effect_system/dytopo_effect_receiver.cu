#include <dytopo_effect_system/dytopo_effect_receiver.h>

namespace uipc::backend::cuda
{
void DyTopoEffectReceiver::do_build()
{
    auto& manager = require<GlobalDyTopoEffectManager>();

    BuildInfo info;
    do_build(info);

    manager.add_receiver(this);
}

void DyTopoEffectReceiver::do_init(InitInfo&) {}

bool DyTopoEffectReceiver::do_accept_raw_full_gradient() const
{
    return false;
}

bool DyTopoEffectReceiver::do_accept_raw_full_hessian() const
{
    return false;
}

void DyTopoEffectReceiver::init()
{
    InitInfo info;
    do_init(info);
}

void DyTopoEffectReceiver::report(GlobalDyTopoEffectManager::ClassifyInfo& info)
{
    do_report(info);

    if constexpr(uipc::RUNTIME_CHECK)
        info.sanity_check();
}
void DyTopoEffectReceiver::receive(GlobalDyTopoEffectManager::ClassifiedDyTopoEffectInfo& info)
{
    do_receive(info);
}

bool DyTopoEffectReceiver::accept_raw_full_hessian() const
{
    return do_accept_raw_full_hessian();
}

bool DyTopoEffectReceiver::accept_raw_full_gradient() const
{
    return do_accept_raw_full_gradient();
}
}  // namespace uipc::backend::cuda
