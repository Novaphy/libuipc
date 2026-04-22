#include <sim_engine.h>
#include <backends/common/module.h>
#include <uipc/common/log.h>

UIPC_BACKEND_API EngineInterface* uipc_create_engine(EngineCreateInfo* info)
{
    uipc::logger::info("[cuda] uipc_create_engine enter");
    return new uipc::backend::cuda::SimEngine(info);
}

UIPC_BACKEND_API void uipc_destroy_engine(EngineInterface* engine)
{
    delete engine;
}