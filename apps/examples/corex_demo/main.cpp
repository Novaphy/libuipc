#include <fmt/core.h>
#include <app/app.h>
#include <app/asset_dir.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/uipc.h>
#include <uipc/builtin/constants.h>
#include <uipc/geometry/utils/affine_body/compute_dyadic_mass.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>

namespace
{
std::optional<int> env_int(const char* key)
{
    if(const char* v = std::getenv(key); v && v[0] != '\0')
        return std::atoi(v);
    return std::nullopt;
}

std::optional<double> env_double(const char* key)
{
    if(const char* v = std::getenv(key); v && v[0] != '\0')
        return std::atof(v);
    return std::nullopt;
}

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

int pick_gpu_device(int argc, char** argv)
{
    int gpu = 0;

    if(const char* env_gpu = std::getenv("UIPC_COREX_GPU_DEVICE");
       env_gpu && env_gpu[0] != '\0')
    {
        gpu = std::max(0, std::atoi(env_gpu));
    }

    for(int i = 1; i < argc; ++i)
    {
        std::string_view a{argv[i]};
        if(a == "--gpu" && i + 1 < argc)
        {
            gpu = std::max(0, std::atoi(argv[i + 1]));
            ++i;
        }
        else if(a.rfind("--gpu=", 0) == 0)
        {
            gpu = std::max(0, std::atoi(std::string{a.substr(6)}.c_str()));
        }
    }

    return gpu;
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
        auto module_dir = std::filesystem::current_path();
        auto release_bin = module_dir / "Release" / "bin";
        if(std::filesystem::exists(release_bin / "libuipc_backend_cuda.so"))
            module_dir = release_bin;
        uipc_config["module_dir"] = module_dir.string();
        uipc::init(uipc_config);
    }

    auto default_output = AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE);
    auto output         = pick_output_dir(argc, argv, default_output);

    auto requested = pick_backend(argc, argv);
    auto frames    = pick_frames(argc, argv);
    auto scene_name = pick_scene(argc, argv);
    auto gpu_device = pick_gpu_device(argc, argv);

    auto engine_config = Engine::default_config();
    engine_config["gpu"]["device"] = gpu_device;
    fmt::println(stderr, "[corex_demo] target gpu device: {}", gpu_device);
    std::fflush(stderr);

    auto make_engine = [&](std::string_view backend) -> Engine
    {
        // Put backend workspace under the per-app output folder.
        return Engine{backend, output, engine_config};
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
    config["line_search"]["max_iter"]       = 64;
    config["newton"]["max_iter"]           = 100;
    // GIPC full mode is performance-oriented: keep the linear solve on the fused
    // device path so SpMV, dot, updates, and convergence checks avoid extra launches.
    config["linear_system"]["solver"]        = "fused_pcg";
    config["linear_system"]["tol_rate"]      = 1e-3;
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
        config["contact"]["friction"]["enable"] = 1;
        // Float/CoreX path: slightly larger activation window reduces missed PT/PE activation.
        config["contact"]["d_hat"]              = 0.03;
        config["sanity_check"]["enable"]        = 1;
        if(auto f = env_int("UIPC_SIMPLE_FORCE_FRICTION_ENABLE"))
            config["contact"]["friction"]["enable"] = (*f != 0) ? 1 : 0;
        if(auto d = env_double("UIPC_SIMPLE_FORCE_DHAT"))
            config["contact"]["d_hat"] = *d;
        if(auto dt = env_double("UIPC_SIMPLE_FORCE_DT"))
            config["dt"] = *dt;
        fmt::println(stderr,
                     "[corex_demo] Corex build: simple — contact/friction/sanity_check ON (d_hat=0.03).");
        std::fflush(stderr);
    }
    else if(scene_name == "slope" || scene_name == "stack" || scene_name == "domino")
    {
        config["contact"]["enable"]             = 1;
        config["contact"]["friction"]["enable"] = 1;
        config["contact"]["d_hat"]              = 0.02;
        config["sanity_check"]["enable"]        = 1;
        if(scene_name == "domino")
        {
            // Initial overlap checks off (pieces start close). Keep d_hat=0.02 like slope/stack (large d_hat is costly).
            config["sanity_check"]["enable"] = 0;
        }
        if(auto d = env_double("UIPC_SCENE_DHAT"))
            config["contact"]["d_hat"] = *d;
        fmt::println(stderr,
                     "[corex_demo] Corex build: {} — contact/friction ON.",
                     scene_name);
        std::fflush(stderr);
    }
#endif

    Scene scene{config};
    {
        AffineBodyConstitution abd;
        scene.constitution_tabular().insert(abd);

        if(scene_name == "simple")
        {
            Float simple_mu    = 0.0;
            Float simple_kappa = 30.0_GPa;
            if(auto mu = env_double("UIPC_SIMPLE_FORCE_MU"))
                simple_mu = static_cast<Float>(*mu);
            if(auto kgpa = env_double("UIPC_SIMPLE_FORCE_KAPPA_GPA"))
                simple_kappa = static_cast<Float>(*kgpa) * 1.0_GPa;
            scene.contact_tabular().default_model(simple_mu, simple_kappa);
            auto default_contact = scene.contact_tabular().default_element();

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
            label_surface(base_mesh);
            label_triangle_orient(base_mesh);

            auto ensure_vertex_contact_id = [](SimplicialComplex& mesh)
            {
                auto meta_cid = mesh.meta().find<IndexT>(builtin::contact_element_id);
                UIPC_ASSERT(meta_cid,
                            "simple scene mesh missing meta contact_element_id after default_element.apply_to()");
                IndexT cid = meta_cid->view().front();

                auto vertex_cid = mesh.vertices().find<IndexT>(builtin::contact_element_id);
                if(!vertex_cid)
                    vertex_cid = mesh.vertices().create<IndexT>(builtin::contact_element_id, cid);
                else
                    std::ranges::fill(view(*vertex_cid), cid);
            };

            SimplicialComplex falling = base_mesh;
            {
                Transform t = Transform::Identity();
                t.translate(Vector3{0.6, 1.5, 0.0});
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

                auto contact_id_attr =
                    falling.meta().find<IndexT>(builtin::contact_element_id);
                fmt::println("[corex_demo:debug] simple falling meta contact_id={}",
                             contact_id_attr ? contact_id_attr->view().front() : -1);
            }

            SimplicialComplex fixed = base_mesh;
            {
                auto is_fixed_attr = fixed.instances().find<IndexT>(builtin::is_fixed);
                view(*is_fixed_attr)[0] = 1;
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

            // Use default contact element for both bodies to keep contact mask enabled.
            // Inter-body filtering is still controlled by body ids / self-collision flags.
            default_contact.apply_to(falling);
            default_contact.apply_to(fixed);
            ensure_vertex_contact_id(falling);
            ensure_vertex_contact_id(fixed);

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

                auto contact_id_attr = fixed.meta().find<IndexT>(builtin::contact_element_id);
                fmt::println("[corex_demo:debug] simple fixed meta contact_id={}",
                             contact_id_attr ? contact_id_attr->view().front() : -1);
            }

            // Keep falling/fixed in separate objects so collision filtering
            // does not treat them as a single self-contact group.
            auto falling_object = scene.objects().create("tets_falling");
            falling_object->geometries().create(falling);
            auto fixed_object = scene.objects().create("tets_fixed");
            fixed_object->geometries().create(fixed);
        }
        else if(scene_name == "slope")
        {
            // --- Inclined plane friction test ---
            std::string tetmesh_dir{AssetDir::tetmesh_path()};

            Float slope_mu    = 0.3;
            Float slope_kappa = 20.0_GPa;
            if(auto mu = env_double("UIPC_SLOPE_MU"))
                slope_mu = static_cast<Float>(*mu);
            scene.contact_tabular().default_model(slope_mu, slope_kappa);
            auto default_contact = scene.contact_tabular().default_element();

            constexpr Float slope_angle_deg = 30.0;
            const Float slope_angle = slope_angle_deg * std::numbers::pi / 180.0;

            // Ramp: a flat slab tilted around Z-axis
            {
                Transform pre = Transform::Identity();
                pre.scale(Vector3{6, 0.3, 3});
                SimplicialComplexIO gio{pre};
                auto ramp = gio.read(fmt::format("{}cube.msh", tetmesh_dir));
                label_surface(ramp);
                label_triangle_orient(ramp);
                abd.apply_to(ramp, 10.0_MPa);
                default_contact.apply_to(ramp);

                Transform t = Transform::Identity();
                t.rotate(AngleAxis(slope_angle, Vector3::UnitZ()));
                t.translate(Vector3{0, 0, 0});
                view(ramp.transforms())[0] = t.matrix();

                auto is_fixed = ramp.instances().find<IndexT>(builtin::is_fixed);
                view(*is_fixed)[0] = 1;
                auto is_dyn = ramp.instances().find<IndexT>(builtin::is_dynamic);
                if(is_dyn) view(*is_dyn)[0] = 0;

                auto ramp_obj = scene.objects().create("ramp");
                ramp_obj->geometries().create(ramp);
            }

            // Slider cube on top of the ramp
            {
                Transform pre = Transform::Identity();
                pre.scale(Vector3{0.5, 0.5, 0.5});
                SimplicialComplexIO gio{pre};
                auto slider = gio.read(fmt::format("{}cube.msh", tetmesh_dir));
                label_surface(slider);
                label_triangle_orient(slider);
                abd.apply_to(slider, 10.0_MPa);
                default_contact.apply_to(slider);

                // Place above ramp surface along the ramp-normal direction
                // Ramp top surface: half-thickness (0.15) along normal from origin
                // Slider half-size: 0.25 along normal + gap
                Float ramp_half_thick = 0.15;
                Float slider_half     = 0.25;
                Float gap             = 0.08;
                Vector3 ramp_normal{-std::sin(slope_angle), std::cos(slope_angle), 0};
                Vector3 pos = ramp_normal * (ramp_half_thick + slider_half + gap);

                Transform t = Transform::Identity();
                t.translate(pos);
                t.rotate(AngleAxis(slope_angle, Vector3::UnitZ()));
                view(slider.transforms())[0] = t.matrix();

                auto is_fixed = slider.instances().find<IndexT>(builtin::is_fixed);
                view(*is_fixed)[0] = 0;
                auto is_dyn = slider.instances().find<IndexT>(builtin::is_dynamic);
                if(is_dyn) view(*is_dyn)[0] = 1;

                auto slider_obj = scene.objects().create("slider");
                slider_obj->geometries().create(slider);
            }

            fmt::println(stderr, "[corex_demo] slope: angle={}°, mu={}", slope_angle_deg, slope_mu);
            std::fflush(stderr);
        }
        else if(scene_name == "stack")
        {
            // --- Stacking stability test ---
            std::string tetmesh_dir{AssetDir::tetmesh_path()};

            Float stack_mu    = 0.5;
            Float stack_kappa = 20.0_GPa;
            if(auto mu = env_double("UIPC_STACK_MU"))
                stack_mu = static_cast<Float>(*mu);
            scene.contact_tabular().default_model(stack_mu, stack_kappa);
            auto default_contact = scene.contact_tabular().default_element();

            // Ground slab
            {
                Transform pre = Transform::Identity();
                pre.scale(Vector3{10, 0.3, 10});
                SimplicialComplexIO gio{pre};
                auto ground = gio.read(fmt::format("{}cube.msh", tetmesh_dir));
                label_surface(ground);
                label_triangle_orient(ground);
                abd.apply_to(ground, 10.0_MPa);
                default_contact.apply_to(ground);

                Transform t = Transform::Identity();
                t.translate(Vector3{0, -0.15, 0});
                view(ground.transforms())[0] = t.matrix();

                auto is_fixed = ground.instances().find<IndexT>(builtin::is_fixed);
                view(*is_fixed)[0] = 1;
                auto is_dyn = ground.instances().find<IndexT>(builtin::is_dynamic);
                if(is_dyn) view(*is_dyn)[0] = 0;

                auto ground_obj = scene.objects().create("ground");
                ground_obj->geometries().create(ground);
            }

            // 3 stacked cubes with slight horizontal offsets
            const Float cube_size = 0.8;
            const Float offsets[] = {0.0, 0.1, -0.15};
            const Float gap       = Float(0.5);
            for(int ci = 0; ci < 3; ci++)
            {
                Transform pre = Transform::Identity();
                pre.scale(Vector3{cube_size, cube_size, cube_size});
                SimplicialComplexIO gio{pre};
                auto cube = gio.read(fmt::format("{}cube.msh", tetmesh_dir));
                label_surface(cube);
                label_triangle_orient(cube);
                abd.apply_to(cube, 10.0_MPa);
                default_contact.apply_to(cube);

                Float y = cube_size * 0.5 + gap + (cube_size + gap) * ci;
                Transform t = Transform::Identity();
                t.translate(Vector3{offsets[ci], y, 0});
                view(cube.transforms())[0] = t.matrix();

                auto is_fixed = cube.instances().find<IndexT>(builtin::is_fixed);
                view(*is_fixed)[0] = 0;
                auto is_dyn = cube.instances().find<IndexT>(builtin::is_dynamic);
                if(is_dyn) view(*is_dyn)[0] = 1;

                auto cube_obj = scene.objects().create(fmt::format("cube_{}", ci));
                cube_obj->geometries().create(cube);
            }

            fmt::println(stderr, "[corex_demo] stack: 3 cubes, gap={}, mu={}", gap, stack_mu);
            std::fflush(stderr);
        }
        else if(scene_name == "domino")
        {
            // --- Domino chain test (tuned for full chain propagation) ---
            std::string tetmesh_dir{AssetDir::tetmesh_path()};

            // Tuned defaults so D1 topples under gravity onto D2; all overridable via env.
            Float domino_mu    = Float(0.25);
            Float domino_kappa = 40.0_GPa;
            Float spacing      = Float(0.55);
            // A 15 degree lean puts D1's center of mass past its front-bottom edge,
            // so gravity tips it forward instead of relying on a horizontal kick.
            Float tilt_deg     = Float(15);
            Float abd_mpa      = Float(1000.0);
            Float d1_vx        = Float(0.0);  // optional +X translational velocity on D1 (m/s)
            // Ground-vs-domino friction, independent of domino-domino friction.
            // Defaults to domino_mu (isotropic). Set UIPC_GROUND_MU to raise it so
            // a struck domino's base sticks and is forced to rotate instead of slide.
            Float ground_mu    = Float(0.8);
            // Optional density override. 1e3 kg/m^3 gives m=80kg per domino; try
            // 100 to verify rotational DoF response (10x lower inertia -> bigger
            // dq_r per Newton step).
            Float domino_density = Float(-1);  // sentinel: "default 1e3"

            if(auto mu = env_double("UIPC_DOMINO_MU"))
                domino_mu = static_cast<Float>(*mu);
            if(auto k = env_double("UIPC_DOMINO_KAPPA_GPA"))
                domino_kappa = static_cast<Float>(*k) * static_cast<Float>(1.0_GPa);
            if(auto s = env_double("UIPC_DOMINO_SPACING"))
                spacing = static_cast<Float>(*s);
            if(auto td = env_double("UIPC_DOMINO_TILT_DEG"))
                tilt_deg = static_cast<Float>(*td);
            if(auto am = env_double("UIPC_DOMINO_ABD_MPA"))
                abd_mpa = static_cast<Float>(*am);
            if(auto vx = env_double("UIPC_DOMINO_VX"))
                d1_vx = static_cast<Float>(*vx);
            if(auto gmu = env_double("UIPC_GROUND_MU"))
                ground_mu = static_cast<Float>(*gmu);
            if(ground_mu < Float(0))
                ground_mu = domino_mu;
            if(auto dd = env_double("UIPC_DOMINO_DENSITY"))
                domino_density = static_cast<Float>(*dd);

            // Default (domino-domino) friction.
            scene.contact_tabular().default_model(domino_mu, domino_kappa);
            auto default_contact = scene.contact_tabular().default_element();
            // Separate element for the ground so we can set an anisotropic
            // ground-vs-domino friction via insert().
            auto ground_contact  = scene.contact_tabular().create("ground");
            scene.contact_tabular().insert(
                ground_contact, default_contact, ground_mu, domino_kappa);

            // Ground slab
            {
                Transform pre = Transform::Identity();
                pre.scale(Vector3{10, 0.2, 4});
                SimplicialComplexIO gio{pre};
                auto ground = gio.read(fmt::format("{}cube.msh", tetmesh_dir));
                label_surface(ground);
                label_triangle_orient(ground);
                abd.apply_to(ground, 10.0_MPa);
                ground_contact.apply_to(ground);

                Transform t = Transform::Identity();
                t.translate(Vector3{2, -0.1, 0});
                view(ground.transforms())[0] = t.matrix();

                auto is_fixed = ground.instances().find<IndexT>(builtin::is_fixed);
                view(*is_fixed)[0] = 1;
                auto is_dyn = ground.instances().find<IndexT>(builtin::is_dynamic);
                if(is_dyn) view(*is_dyn)[0] = 0;

                auto ground_obj = scene.objects().create("ground");
                ground_obj->geometries().create(ground);
            }

            const int     num_dominos = 5;
            const Float   half_h      = Float(0.5);
            const Vector3 domino_scale{Float(0.2), Float(1.0), Float(0.4)};
            const Float   gap = Float(0.05);

            for(int di = 0; di < num_dominos; di++)
            {
                Transform pre = Transform::Identity();
                pre.scale(domino_scale);
                SimplicialComplexIO gio{pre};
                auto domino = gio.read(fmt::format("{}cube.msh", tetmesh_dir));
                label_surface(domino);
                label_triangle_orient(domino);
                if(domino_density > Float(0))
                    abd.apply_to(domino,
                                 abd_mpa * static_cast<Float>(1.0_MPa),
                                 domino_density);
                else
                    abd.apply_to(domino, abd_mpa * static_cast<Float>(1.0_MPa));
                default_contact.apply_to(domino);

                Float x = di * spacing;
                Transform t = Transform::Identity();
                if(di == 0 && tilt_deg > Float(0))
                {
                    // Tilt D1 about its front-bottom edge so gravity tips it forward.
                    const Float   half_w   = domino_scale.x() * Float(0.5);
                    const Vector3 pivot_local{half_w, -half_h, Float(0)};
                    const Vector3 world_pivot{x + half_w, gap, Float(0)};
                    Float         tilt_rad =
                        tilt_deg * Float(std::numbers::pi / 180.0);
                    t.translate(world_pivot);
                    t.rotate(AngleAxis(-tilt_rad, Vector3::UnitZ()));
                    t.translate(-pivot_local);
                }
                else
                {
                    t.translate(Vector3{x, half_h + gap, Float(0)});
                }

                view(domino.transforms())[0] = t.matrix();

                auto is_fixed = domino.instances().find<IndexT>(builtin::is_fixed);
                view(*is_fixed)[0] = 0;
                auto is_dyn = domino.instances().find<IndexT>(builtin::is_dynamic);
                if(is_dyn) view(*is_dyn)[0] = 1;

                // Seed D1 with an initial +X linear velocity to "nudge" the chain.
                // abd.apply_to already allocates the builtin::velocity <Matrix4x4> attribute.
                if(di == 0 && d1_vx != Float(0))
                {
                    auto vel_attr =
                        domino.instances().find<Matrix4x4>(builtin::velocity);
                    if(vel_attr)
                    {
                        Matrix4x4 vel   = Matrix4x4::Zero();
                        vel(0, 3)       = d1_vx;
                        view(*vel_attr)[0] = vel;
                    }
                }

                auto domino_obj = scene.objects().create(fmt::format("domino_{}", di));
                domino_obj->geometries().create(domino);
            }

            fmt::println(stderr,
                         "[corex_demo] domino: {} pieces, 0.2x1.0x0.4, spacing={}, tilt={}deg, abd={}MPa, kappa={}GPa, mu_d={}, mu_g={}, d1_vx={}m/s",
                         num_dominos,
                         spacing,
                         tilt_deg,
                         abd_mpa,
                         domino_kappa / static_cast<Float>(1.0_GPa),
                         domino_mu,
                         ground_mu,
                         d1_vx);
            std::fflush(stderr);
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

        const bool profile_all_frames = std::getenv("UIPC_COREX_PHASE_PROFILE") != nullptr;
        if(profile_all_frames || i <= 3 || i == frames - 1)
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
