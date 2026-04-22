#include <catch2/catch_all.hpp>
#include <uipc/common/logger.h>
#include <uipc/common/uipc.h>

#include <algorithm>
#include <filesystem>
#include <string>

namespace
{
uipc::Logger::Level parse_log_level(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c)
                   { return static_cast<char>(std::tolower(c)); });

    if(value == "trace")
        return spdlog::level::trace;
    if(value == "debug")
        return spdlog::level::debug;
    if(value == "info")
        return spdlog::level::info;
    if(value == "warn" || value == "warning")
        return spdlog::level::warn;
    if(value == "error")
        return spdlog::level::err;
    if(value == "critical")
        return spdlog::level::critical;
    if(value == "off")
        return spdlog::level::off;

    // Keep default if unknown.
    return uipc::logger::get_level();
}
}  // namespace

int main(int argc, char* argv[])
{
    Catch::Session session;

    std::string log_level;
    auto cli = session.cli() | Catch::Clara::Opt(log_level, "level")["--log-level"](
                                   "Set logger level: trace|debug|info|warn|error|critical|off");

    session.cli(cli);

    const int result = session.applyCommandLine(argc, argv);
    if(result != 0)
        return result;

    if(!log_level.empty())
        uipc::logger::set_level(parse_log_level(log_level));

    // Tests that instantiate uipc::core::Engine/SanityChecker need module_dir
    // to point at the directory holding libuipc_backend_*.so. Mirror the
    // resolution used by examples/corex_demo/main.cpp so tests work whether
    // launched from the repo root or from build_*/Release/bin.
    {
        namespace fs    = std::filesystem;
        auto uipc_cfg   = uipc::default_config();
        auto module_dir = fs::current_path();
        auto release_bin = module_dir / "Release" / "bin";
        if(fs::exists(release_bin / "libuipc_backend_none.so")
           || fs::exists(release_bin / "libuipc_backend_cuda.so"))
        {
            module_dir = release_bin;
        }
        uipc_cfg["module_dir"] = module_dir.string();
        try
        {
            uipc::init(uipc_cfg);
        }
        catch(const std::exception& e)
        {
            uipc::logger::warn("uipc::init failed in test main: {}", e.what());
        }
    }

    return session.run();
}
