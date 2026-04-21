#include <uipc/core/engine.h>
#include <uipc/core/internal/engine.h>
#include <dylib.hpp>
#include <uipc/backend/module_init_info.h>
#include <uipc/backend/engine_create_info.h>
#include <uipc/common/uipc.h>
#include <filesystem>
#include <uipc/common/log_pattern_guard.h>
#include <fstream>

namespace uipc::core::internal
{
using DylibModule = ::dylib::library;
namespace fs = std::filesystem;

static S<DylibModule> load_dylib_by_dir_and_name(const std::string& module_dir,
                                                 const std::string& module_name)
{
    fs::path module_path = fs::path(module_dir) / module_name;
    uipc::logger::info("[core] loading dylib {}", module_path.string());
    return uipc::make_shared<DylibModule>(module_path.string(),
                                          ::dylib::decorations::os_default());
}

static string to_lower(std::string_view s)
{
    string result{s};
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

class Engine::Impl
{
    using Deleter = void (*)(IEngine*);

    string   m_backend_name;
    S<DylibModule> m_module;

    IEngine*   m_engine    = nullptr;
    S<IEngine> m_overrider = nullptr;

    Deleter      m_deleter   = nullptr;
    mutable bool m_sync_flag = false;
    string       m_workspace;

    static unordered_map<string, S<DylibModule>> m_cache;
    static std::mutex                      m_cache_mutex;

    static S<DylibModule> load_module(std::string_view backend_name)
    {
        std::lock_guard lock{m_cache_mutex};

        // find in cache
        auto it = m_cache.find(std::string{backend_name});
        if(it != m_cache.end())
            return it->second;

        // if not found, load it
        auto& uipc_config = uipc::config();
        uipc::logger::info("[core] load_module({}): before dlopen", backend_name);
        auto backend = load_dylib_by_dir_and_name(uipc_config["module_dir"].get<std::string>(),
                                                  fmt::format("uipc_backend_{}", backend_name));
        uipc::logger::info("[core] load_module({}): after dlopen", backend_name);

        auto info             = make_unique<UIPCModuleInitInfo>();
        info->module_name     = backend_name;
        info->memory_resource = std::pmr::get_default_resource();

        try
        {
            uipc::logger::info("[core] load_module({}): before uipc_init_module", backend_name);
            auto init = backend->get_function<void(UIPCModuleInitInfo*)>("uipc_init_module");
            init(info.get());
            uipc::logger::info("[core] load_module({}): after uipc_init_module", backend_name);
        }
        catch(const dylib::symbol_error&)
        {
            throw Exception{fmt::format("Can't find backend [{}]'s module initializer.",
                                        backend_name)};
        }

        return m_cache[std::string{backend_name}] = backend;
    }

  public:
    Impl(std::string_view backend_name, std::string_view workspace, const Json& config)
        : m_backend_name(to_lower(backend_name))
    {
        auto& uipc_config = uipc::config();

        uipc::logger::info("[core] Engine::Impl ctor: before load_module({})", m_backend_name);
        m_module = load_module(m_backend_name);
        uipc::logger::info("[core] Engine::Impl ctor: after load_module({})", m_backend_name);

        m_workspace = fs::absolute(workspace).string();
        if(!fs::exists(m_workspace))  // create workspace
        {
            fs::create_directory(m_workspace);
        }
        else if(!fs::is_directory(m_workspace))
        {
            throw EngineException{fmt::format("Workspace [{}] is not a directory.", workspace)};
        }

        EngineCreateInfo info;
        info.workspace = m_workspace;
        info.config    = config;
        try
        {
            uipc::logger::info("[core] Engine::Impl ctor: before uipc_create_engine");
            auto creator = m_module->get_function<IEngine*(EngineCreateInfo*)>("uipc_create_engine");
            {
                LogPatternGuard guard{backend_name};
                m_engine = creator(&info);
            }
            uipc::logger::info("[core] Engine::Impl ctor: after uipc_create_engine");
            m_deleter = m_module->get_function<void(IEngine*)>("uipc_destroy_engine");
        }
        catch(const dylib::symbol_error&)
        {
            throw EngineException{
                fmt::format("Can't find backend [{}]'s engine entrypoints (create/destroy).",
                            backend_name)};
        }
    }

    Impl(std::string_view backend_name, S<IEngine> overrider, std::string_view workspace, const Json& config)
        : m_backend_name(to_lower(backend_name))
        , m_module(nullptr)  // no module for overrider
        , m_overrider(overrider)
        , m_engine(overrider.get())
        , m_deleter{nullptr}
        , m_workspace(workspace)
    {
        if(!m_engine)
            throw EngineException{"Overrider engine is null."};
    }

    std::string_view backend_name() const noexcept { return m_backend_name; }

    void init(internal::World& w)
    {
        LogPatternGuard guard{backend_name()};
        m_engine->init(w);
    }

    void advance()
    {
        m_sync_flag = false;
        LogPatternGuard guard{backend_name()};
        m_engine->advance();
    }

    void sync()
    {
        LogPatternGuard guard{backend_name()};
        m_engine->sync();
        m_sync_flag = true;
    }

    void retrieve()
    {
        LogPatternGuard guard{backend_name()};
        if(!m_sync_flag)
            m_engine->sync();
        m_engine->retrieve();
    }

    bool do_dump()
    {
        LogPatternGuard guard{backend_name()};
        return m_engine->dump();
    }

    bool do_recover(SizeT dst_frame)
    {
        LogPatternGuard guard{backend_name()};
        return m_engine->recover(dst_frame);
    }

    SizeT get_frame() const
    {
        LogPatternGuard guard{backend_name()};
        return m_engine->frame();
    }

    Json to_json() const
    {
        Json j;
        // force copy
        j = m_engine->to_json();
        return j;
    }

    EngineStatusCollection& status()
    {
        LogPatternGuard guard{backend_name()};
        return m_engine->status();
    }

    const FeatureCollection& features()
    {
        LogPatternGuard guard{backend_name()};
        return m_engine->features();
    }

    std::string_view workspace() const noexcept { return m_workspace; }

    ~Impl()
    {
        if(!m_overrider && m_deleter && m_engine)
        {
            m_deleter(m_engine);
        }
    }
};

unordered_map<string, S<DylibModule>> Engine::Impl::m_cache;
std::mutex                      Engine::Impl::m_cache_mutex;


Engine::Engine(std::string_view backend_name, std::string_view workspace, const Json& config)
    : m_impl{uipc::make_unique<Impl>(backend_name, workspace, config)}
{
}

Engine::Engine(std::string_view backend_name,
               S<IEngine>       overrider,
               std::string_view workspace,
               const Json&      config)
    : m_impl{uipc::make_unique<Impl>(backend_name, std::move(overrider), workspace, config)}
{
}

Engine::~Engine() {}

Json Engine::default_config()
{
    Json j;
    auto config_path      = std::getenv("UIPC_ENGINE_DEFAULT_CONFIG");
    bool override_default = false;

    if(config_path)
    {
        std::ifstream ifs{config_path};
        if(!ifs)
        {
            logger::warn("Load default config file [{}] failed, fallback to default config.",
                         config_path);
        }
        else
        {
            try
            {
                j                = Json::parse(ifs);
                override_default = true;
                logger::info("Override default config using config file [{}] successfully.",
                             config_path);
            }
            catch(const std::exception& e)
            {
                logger::warn("Load default config file [{}] failed: {}, rollback to default config.",
                             config_path,
                             e.what());
            }
        }
    }

    if(!override_default)
    {
        j["gpu"]["device"]           = 0;
        j["extras"]["gui"]["enable"] = true;
    }
    return j;
}

std::string_view Engine::backend_name() const noexcept
{
    return m_impl->backend_name();
}

std::string_view Engine::workspace() const noexcept
{
    return m_impl->workspace();
}

EngineStatusCollection& Engine::status()
{
    return m_impl->status();
}

const FeatureCollection& Engine::features()
{
    return m_impl->features();
}

void Engine::init(internal::World& v)
{
    m_impl->init(v);
}

void Engine::advance()
{
    m_impl->advance();
}

void Engine::sync()
{
    m_impl->sync();
}

void Engine::retrieve()
{
    m_impl->retrieve();
}

Json Engine::to_json() const
{
    return m_impl->to_json();
}
bool Engine::dump()
{
    return m_impl->do_dump();
}
bool Engine::recover(SizeT dst_frame)
{
    return m_impl->do_recover(dst_frame);
}
SizeT Engine::frame() const
{
    return m_impl->get_frame();
}
}  // namespace uipc::core::internal
