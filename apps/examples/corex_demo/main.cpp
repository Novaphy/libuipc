#include <fmt/core.h>
#include <app/app.h>
#include <app/asset_dir.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/uipc.h>
#include <uipc/builtin/constants.h>
#include <uipc/geometry/utils/affine_body/compute_dyadic_mass.h>

#include <cstdio>
#include <fstream>
#include <filesystem>
#include <numbers>
#include <string>
#include <string_view>

namespace
{
std::string_view pick_backend(int argc, char** argv)
{
    // Default: try cuda first.
    std::string_view backend = "cuda";

    for(int i = 1; i < argc; ++i)
    {
        std::string_view a{argv[i]};
        if(a == "--backend" && i + 1 < argc)
        {
            backend = argv[i + 1];
            ++i;
        }
        else if(a.rfind("--backend=", 0) == 0)
        {
            backend = a.substr(std::string_view{"--backend="}.size());
        }
    }
    return backend;
}

int pick_frames(int argc, char** argv)
{
    int frames = 50;

    for(int i = 1; i < argc; ++i)
    {
        std::string_view a{argv[i]};
        if(a == "--frames" && i + 1 < argc)
        {
            frames = std::max(1, std::atoi(argv[i + 1]));
            ++i;
        }
        else if(a.rfind("--frames=", 0) == 0)
        {
            frames = std::max(1, std::atoi(std::string{a.substr(9)}.c_str()));
        }
    }
    return frames;
}

std::string pick_output_dir(int argc, char** argv, const std::string& default_output)
{
    // Accept either:
    //   --output_dir <dir>   / --output_dir=<dir>
    //   --output <dir>       / --output=<dir>
    // to override AssetDir::output_path(...).
    std::string out = default_output;

    for(int i = 1; i < argc; ++i)
    {
        std::string_view a{argv[i]};
        auto pick_next = [&](const char* key) -> bool
        {
            if(a == key && i + 1 < argc)
            {
                out = argv[i + 1];
                ++i;
                return true;
            }
            return false;
        };

        if(pick_next("--output_dir") || pick_next("--output"))
            continue;

        if(a.rfind("--output_dir=", 0) == 0)
            out = std::string{a.substr(std::string_view{"--output_dir="}.size())};
        else if(a.rfind("--output=", 0) == 0)
            out = std::string{a.substr(std::string_view{"--output="}.size())};
    }

    // Normalize: ensure it ends with a directory separator so fmt::format("{}file", out) is safe.
    if(!out.empty())
    {
        std::filesystem::path p{out};
        // If user passed a path without trailing slash, we still want "dir/" semantics.
        // We can't reliably detect whether it's file vs dir, so assume dir.
        out = (p / "").string();
    }
    return out;
}

std::string_view pick_scene(int argc, char** argv)
{
#if defined(UIPC_APP_COREX_BUILD) && UIPC_APP_COREX_BUILD
    std::string_view scene = "simple";
#else
    std::string_view scene = "wrecking_ball";
#endif

    for(int i = 1; i < argc; ++i)
    {
        std::string_view a{argv[i]};
        if(a == "--scene" && i + 1 < argc)
        {
            scene = argv[i + 1];
            ++i;
        }
        else if(a.rfind("--scene=", 0) == 0)
        {
            scene = a.substr(std::string_view{"--scene="}.size());
        }
    }
    return scene;
}
}  // namespace

int main(int argc, char** argv)
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;

    logger::set_level(spdlog::level::info);

    // Explicitly initialize module_dir so backend dylibs are loadable.
    // This avoids relying on implicit defaults that may be invalid on some runtimes.
    {
        auto uipc_config = uipc::default_config();
        uipc_config["module_dir"] = std::filesystem::current_path().string();
        uipc::init(uipc_config);
    }

    auto default_output = AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE);
    auto output         = pick_output_dir(argc, argv, default_output);

    auto requested = pick_backend(argc, argv);
    auto frames    = pick_frames(argc, argv);
    auto scene_name = pick_scene(argc, argv);

    auto make_engine = [&](std::string_view backend) -> Engine
    {
        // Put backend workspace under the per-app output folder.
        return Engine{backend, output};
    };

    Engine engine = [&]() -> Engine
    {
        try
        {
            return make_engine(requested);
        }
        catch(const EngineException& e)
        {
            if(requested != "none")
            {
                fmt::println("Failed to start backend '{}': {}", requested, e.what());
                fmt::println("Falling back to backend 'none' for a smoke-test.");
                return make_engine("none");
            }
            throw;
        }
    }();

    fmt::println("Using backend: {}", engine.backend_name());
    World world{engine};

    auto config      = Scene::default_config();
    config["dt"]     = 0.01_s;
    config["gravity"] = Vector3{0, -9.8, 0};
    config["contact"]["enable"]             = true;
    config["contact"]["friction"]["enable"] = true;
    config["contact"]["d_hat"]              = 0.01;
    config["line_search"]["max_iter"]       = 8;
    // Limit iterations for debug dumps.
    config["newton"]["max_iter"]           = 4;
    // Corex: prefer the non-fused PCG path for stability/compatibility.
    // (fused_pcg uses a more aggressive fused-kernel implementation that may stall on some CUDA-compat runtimes)
    config["linear_system"]["solver"]        = "linear_pcg";
    config["linear_system"]["tol_rate"]      = 1e-2;
    config["linear_system"]["check_interval"] = 1;
    config["sanity_check"]["enable"]       = 1;
    // Dump linear system to check whether the solver is producing updates.
    config["extras"]["debug"]["dump_linear_system"] = 0;
    // Keep surface dumping off here to avoid huge files.
    config["extras"]["debug"]["dump_surface"]        = 0;
    config["collision_detection"]["method"] = "stackless_bvh";

#if defined(UIPC_APP_COREX_BUILD) && UIPC_APP_COREX_BUILD

    if(scene_name == "simple")
    {
        config["contact"]["enable"]             = 1;
        config["contact"]["friction"]["enable"] = 0;
        config["sanity_check"]["enable"] = 0;
        fmt::println(stderr,
                     "[corex_demo] Corex build: simple — contact ON, friction off, sanity_check off.");
        std::fflush(stderr);
    }
#endif

    Scene scene{config};
    {
        AffineBodyConstitution abd;
        scene.constitution_tabular().insert(abd);

        if(scene_name == "simple")
        {
            scene.contact_tabular().default_model(0.5, 1.0_GPa);
            auto default_element = scene.contact_tabular().default_element();

            vector<Vector3> Vs = {Vector3{0, 1, 0},
                                  Vector3{0, 0, 1},
                                  Vector3{-std::sqrt(3) / 2, 0, -0.5},
                                  Vector3{std::sqrt(3) / 2, 0, -0.5}};
            vector<Vector4i> Ts = {Vector4i{0, 1, 2, 3}};

            SimplicialComplex base_mesh = tetmesh(Vs, Ts);
            abd.apply_to(base_mesh, 100.0_MPa);
            // Debug: ensure mass-related attributes are non-zero.
            {
                auto md_attr = base_mesh.meta().find<Float>(builtin::mass_density);
                auto vol_attr = base_mesh.meta().find<Float>(builtin::volume);
                if(md_attr && vol_attr)
                {
                    fmt::println("[corex_demo:debug] base_mesh mass_density={}, volume={}",
                                 md_attr->view().front(),
                                 vol_attr->view().front());

                    // Host-side dyadic mass sanity check (should be non-zero).
                    Float   rho = md_attr->view().front();
                    Float   m   = 0.0;
                    Vector3 mx_bar;
                    Matrix3x3 mxx;
                    uipc::geometry::affine_body::compute_dyadic_mass(base_mesh,
                                                                       rho,
                                                                       m,
                                                                       mx_bar,
                                                                       mxx);
                    fmt::println("[corex_demo:debug] host dyadic mass m={}",
                                 m);
                }
                else
                {
                    fmt::println("[corex_demo:debug] base_mesh missing mass_density/volume attr");
                }
            }
            default_element.apply_to(base_mesh);
            label_surface(base_mesh);
            label_triangle_orient(base_mesh);

            SimplicialComplex falling = base_mesh;
            {
                // For affine-body motion, the backend streams instance transforms.
                // Apply the initial offset via transforms (not by mutating positions),
                // so that subsequent time integration has meaningful translation DOFs.
                Transform t = Transform::Identity();
                t.translate(Vector3::UnitY() * 1.005);
                view(falling.transforms())[0] = t.matrix();
            }
            // Explicitly mark falling body as dynamic (not fixed).
            {
                auto is_fixed_attr   = falling.instances().find<IndexT>(builtin::is_fixed);
                auto is_dynamic_attr = falling.instances().find<IndexT>(builtin::is_dynamic);
                if(is_fixed_attr)
                    view(*is_fixed_attr)[0] = 0;
                if(is_dynamic_attr)
                    view(*is_dynamic_attr)[0] = 1;

                // Debug: verify instance flags before init.
                if(is_fixed_attr && is_dynamic_attr)
                {
                    fmt::println("[corex_demo:debug] simple falling is_fixed={}, is_dynamic={}",
                                 view(*is_fixed_attr)[0],
                                 view(*is_dynamic_attr)[0]);
                }
            }

            // Debug: confirm tetrahedra topology is still present.
            {
                auto tets = falling.tetrahedra().topo().view();
                fmt::println("[corex_demo:debug] simple falling tetra_count={}", tets.size());
                fmt::println("[corex_demo:debug] simple falling positions_count={}",
                             falling.positions().size());

                auto md_attr = falling.meta().find<Float>(builtin::mass_density);
                auto vol_attr = falling.meta().find<Float>(builtin::volume);
                fmt::println("[corex_demo:debug] simple falling meta mass_density={}, volume={}",
                             md_attr ? md_attr->view().front() : -1.0,
                             vol_attr ? vol_attr->view().front() : -1.0);
            }

            SimplicialComplex fixed = base_mesh;
            {
                auto is_fixed_attr = fixed.instances().find<IndexT>(builtin::is_fixed);
                view(*is_fixed_attr)[0] = 1;
                // A fixed body shouldn't participate in kinetics.
                auto is_dynamic_attr = fixed.instances().find<IndexT>(builtin::is_dynamic);
                if(is_dynamic_attr)
                    view(*is_dynamic_attr)[0] = 0;

                // Debug: verify instance flags before init.
                if(is_fixed_attr && is_dynamic_attr)
                {
                    fmt::println("[corex_demo:debug] simple fixed is_fixed={}, is_dynamic={}",
                                 view(*is_fixed_attr)[0],
                                 view(*is_dynamic_attr)[0]);
                }
            }

            {
                auto tets = fixed.tetrahedra().topo().view();
                fmt::println("[corex_demo:debug] simple fixed tetra_count={}", tets.size());
                fmt::println("[corex_demo:debug] simple fixed positions_count={}",
                             fixed.positions().size());

                auto md_attr = fixed.meta().find<Float>(builtin::mass_density);
                auto vol_attr = fixed.meta().find<Float>(builtin::volume);
                fmt::println("[corex_demo:debug] simple fixed meta mass_density={}, volume={}",
                             md_attr ? md_attr->view().front() : -1.0,
                             vol_attr ? vol_attr->view().front() : -1.0);
            }

            auto object = scene.objects().create("tets");
            object->geometries().create(falling);
            object->geometries().create(fixed);
        }
        else  // wrecking_ball
        {
            std::string tetmesh_dir{AssetDir::tetmesh_path()};
            // Reuse the official example's scene JSON (keeps demo self-contained without copying assets).
            auto this_folder =
                AssetDir::folder("apps/examples/wrecking_ball/main.cpp");

            Json wrecking_ball_scene;
            {
                std::ifstream ifs(fmt::format("{}wrecking_ball.json", this_folder));
                ifs >> wrecking_ball_scene;
            }

            auto default_contact = scene.contact_tabular().default_element();
            scene.contact_tabular().default_model(0.01, 20.0_GPa);

            Float     scale = 1;
            Transform T     = Transform::Identity();
            T.scale(scale);
            SimplicialComplexIO io{T};

            auto cube = io.read(fmt::format("{}cube.msh", tetmesh_dir));
            auto ball = io.read(fmt::format("{}ball.msh", tetmesh_dir));
            auto link = io.read(fmt::format("{}link.msh", tetmesh_dir));

            S<Object> cube_obj = scene.objects().create("cubes");
            S<Object> ball_obj = scene.objects().create("balls");
            S<Object> link_obj = scene.objects().create("links");

            abd.apply_to(cube, 10.0_MPa);
            label_surface(cube);
            label_triangle_orient(cube);
            abd.apply_to(ball, 10.0_MPa);
            label_surface(ball);
            label_triangle_orient(ball);
            abd.apply_to(link, 10.0_MPa);
            label_surface(link);
            label_triangle_orient(link);

            default_contact.apply_to(cube);
            default_contact.apply_to(ball);
            default_contact.apply_to(link);

            auto build_mesh = [&](const Json& j, Object& obj, const SimplicialComplex& mesh)
            {
                Vector3 position;

                if(j.find("position") != j.end())
                {
                    position[0] = j["position"][0].get<Float>();
                    position[1] = j["position"][1].get<Float>();
                    position[2] = j["position"][2].get<Float>();
                }

                Eigen::Quaternion<Float> Q = Eigen::Quaternion<Float>::Identity();

                if(j.find("rotation") != j.end())
                {
                    Vector3 rotation;
                    rotation[0] = j["rotation"][0].get<Float>();
                    rotation[1] = j["rotation"][1].get<Float>();
                    rotation[2] = j["rotation"][2].get<Float>();

                    rotation *= std::numbers::pi / 180.0;

                    Q = AngleAxis(rotation.z(), Vector3::UnitZ())
                        * AngleAxis(rotation.y(), Vector3::UnitY())
                        * AngleAxis(rotation.x(), Vector3::UnitX());
                }

                IndexT is_fixed = 0;
                if(j.find("is_dof_fixed") != j.end())
                {
                    is_fixed = j["is_dof_fixed"].get<bool>() ? 1 : 0;
                }

                position *= scale;

                Transform t = Transform::Identity();
                t.translate(position).rotate(Q);

                SimplicialComplex this_mesh     = mesh;
                view(this_mesh.transforms())[0] = t.matrix();

                auto is_fixed_attr = this_mesh.instances().find<IndexT>(builtin::is_fixed);
                view(*is_fixed_attr)[0] = is_fixed;
                // Ensure dynamics is consistent with the fixed flag.
                // Otherwise some runtimes / copy paths may keep bodies kinematic,
                // causing q (and thus transforms) to remain unchanged across frames.
                auto is_dynamic_attr =
                    this_mesh.instances().find<IndexT>(builtin::is_dynamic);
                if(is_dynamic_attr)
                    view(*is_dynamic_attr)[0] = is_fixed ? 0 : 1;

                obj.geometries().create(this_mesh);
            };

            for(const Json& obj : wrecking_ball_scene)
            {
                if(obj["mesh"] == "link.msh")
                    build_mesh(obj, *link_obj, link);
                else if(obj["mesh"] == "cube.msh")
                    build_mesh(obj, *cube_obj, cube);
                else if(obj["mesh"] == "ball.msh")
                    build_mesh(obj, *ball_obj, ball);
            }

            // Ground: use a mesh ground to avoid half-plane TOI corner cases on some runtimes.
            {
                Transform pre_transform = Transform::Identity();
                pre_transform.scale(Vector3{40, 0.2, 40});

                SimplicialComplexIO gio{pre_transform};
                auto               ground = gio.read(fmt::format("{}{}", tetmesh_dir, "cube.msh"));

                label_surface(ground);
                label_triangle_orient(ground);

                Transform transform = Transform::Identity();
                transform.translate(Vector3{12, -1.1, 0});
                view(ground.transforms())[0] = transform.matrix();

                abd.apply_to(ground, 10.0_MPa);

                auto is_fixed = ground.instances().find<IndexT>(builtin::is_fixed);
                view(*is_fixed)[0] = 1;

                auto ground_obj = scene.objects().create("ground");
                ground_obj->geometries().create(ground);
            }
        }
    }

    test::Scene::dump_config(config, output);
    fmt::println(stderr, "[corex_demo] world.init(scene) ... (first CUDA init may take a while due to JIT)");
    std::fflush(stderr);
    world.init(scene);
    if(!world.is_valid())
    {
        fmt::println(stderr, "[corex_demo] ERROR: world.init failed (invalid world). Check logs above.");
        std::fflush(stderr);
        return 1;
    }
    fmt::println(stderr, "[corex_demo] world.init OK");
    std::fflush(stderr);

    SceneIO sio{scene};
    sio.write_surface(fmt::format("{}scene_surface_{:04d}.obj", output, 0));

    if(engine.backend_name() == "none")
        frames = 1;

    for(int i = 1; i < frames; ++i)
    {
#if defined(UIPC_APP_COREX_BUILD) && UIPC_APP_COREX_BUILD
        if(i <= 3 || i == frames - 1)
        {
            fmt::println(stderr, "[corex_demo] frame {} / {} ...", i, frames);
            std::fflush(stderr);
        }
#endif
        auto t0 = std::chrono::steady_clock::now();
        world.advance();
        auto t1 = std::chrono::steady_clock::now();
        world.sync();
        auto t2 = std::chrono::steady_clock::now();
        world.retrieve();
        auto t3 = std::chrono::steady_clock::now();
        sio.write_surface(fmt::format("{}scene_surface_{:04d}.obj", output, i));
        auto t4 = std::chrono::steady_clock::now();

        if(i <= 3 || i == frames - 1)
        {
            auto ms = [](auto a, auto b)
            {
                return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
            };
            fmt::println(stderr,
                         "[corex_demo] frame {} timings: advance={}ms sync={}ms retrieve={}ms write_obj={}ms",
                         i,
                         ms(t0, t1),
                         ms(t1, t2),
                         ms(t2, t3),
                         ms(t3, t4));
            std::fflush(stderr);
        }
    }

    fmt::println("Wrote OBJ sequence to: {}", output);
    return 0;
}

