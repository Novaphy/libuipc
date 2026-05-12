#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
#pragma once
#include <uipc/common/dllexport.h>
#include <uipc/backend/visitors/world_visitor.h>
#include <uipc/core/engine_status.h>
#include <uipc/core/feature_collection.h>
#include <uipc/core/i_sanity_checker.h>

namespace uipc::core
{
class World;

class UIPC_CORE_API IEngine
{
  public:
    virtual ~IEngine() = default;
    void init(internal::World& w);
    void advance();
    void sync();
    void retrieve();
    Json to_json() const;

    bool                     dump();
    bool                     recover(SizeT dst_frame);
    SizeT                    frame() const;
    EngineStatusCollection&  status();
    const FeatureCollection& features() const;

    /**
     * @brief Insert backend-specific sanity checkers into @p collection.
     *
     * Engines that ship their own sanity checkers (e.g. the cuda backend)
     * override do_insert_sanity_checkers(); the default implementation is
     * a no-op so backends without sanity checkers don't need to override.
     */
    void insert_sanity_checkers(ISanityCheckerCollection& collection);

  protected:
    virtual void                     do_init(internal::World&) = 0;
    virtual void                     do_advance()              = 0;
    virtual void                     do_sync()                 = 0;
    virtual void                     do_retrieve()             = 0;
    virtual Json                     do_to_json() const;
    virtual bool                     do_dump();
    virtual bool                     do_recover(SizeT dst_frame);
    virtual SizeT                    get_frame() const    = 0;
    virtual EngineStatusCollection&  get_status()         = 0;
    virtual const FeatureCollection& get_features() const = 0;
    virtual void do_insert_sanity_checkers(ISanityCheckerCollection& collection);
};
}  // namespace uipc::core
#else
#pragma once
#include <uipc/common/dllexport.h>
#include <uipc/backend/visitors/world_visitor.h>
#include <uipc/core/engine_status.h>
#include <uipc/core/feature_collection.h>
namespace uipc::core
{
class World;
class ISanityCheckerCollection;

class UIPC_CORE_API IEngine
{
  public:
    virtual ~IEngine() = default;
    void init(internal::World& w);
    void advance();
    void sync();
    void retrieve();
    Json to_json() const;

    bool                     dump();
    bool                     recover(SizeT dst_frame);
    SizeT                    frame() const;
    EngineStatusCollection&  status();
    const FeatureCollection& features() const;

    void insert_sanity_checkers(ISanityCheckerCollection& collection);

  protected:
    virtual void                     do_init(internal::World&) = 0;
    virtual void                     do_advance()              = 0;
    virtual void                     do_sync()                 = 0;
    virtual void                     do_retrieve()             = 0;
    virtual Json                     do_to_json() const;
    virtual bool                     do_dump();
    virtual bool                     do_recover(SizeT dst_frame);
    virtual SizeT                    get_frame() const    = 0;
    virtual EngineStatusCollection&  get_status()         = 0;
    virtual const FeatureCollection& get_features() const = 0;
    virtual void do_insert_sanity_checkers(ISanityCheckerCollection& collection);
};
}  // namespace uipc::core
#endif
