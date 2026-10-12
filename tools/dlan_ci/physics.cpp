#include <uipc/uipc.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/constitution/affine_body_external_force.h>
#include <uipc/core/affine_body_state_accessor_feature.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
using namespace uipc::constitution;

static_assert(sizeof(Float) == 4, "DLAN physics CI requires FP32");

namespace
{
void require(bool condition, const std::string& message)
{
    if(!condition)
        throw std::runtime_error(message);
}

SimplicialComplex cube(Float mass, const Vector3& position)
{
    Transform scale = Transform::Identity();
    scale.scale(Float(0.3));
    SimplicialComplexIO io{scale};
    auto mesh = io.read(std::string(UIPC_DLAN_SOURCE_DIR) + "/assets/sim_data/tetmesh/cube.msh");
    AffineBodyConstitution{}.apply_to(mesh, Float(1e8), mass / Float(0.027));
    label_surface(mesh);
    label_triangle_orient(mesh);
    Transform pose             = Transform::Identity();
    pose.translation()         = position;
    view(mesh.transforms())[0] = pose.matrix();
    return mesh;
}

void advance(World& world)
{
    world.advance();
    world.sync();
    require(world.is_valid(), "GPU simulation became invalid");
    world.retrieve();
}

void run(const std::string& name, Float mass)
{
    const auto workspace = std::filesystem::current_path() / "physics-output"
                           / (name + "-" + std::to_string(mass));
    std::filesystem::create_directories(workspace);
    Engine engine{"cuda", workspace.string() + "/"};
    require(engine.backend_name() == "cuda", "GPU backend fallback is forbidden");
    World           world{engine};
    constexpr Float dt                = Float(0.01);
    const bool      stack             = name == "stack";
    const Vector3   gravity           = (name == "free-fall" || stack) ?
                                            Vector3{0, Float(-9.81), 0} :
                                            Vector3::Zero().eval();
    auto            config            = Scene::default_config();
    config["dt"]                      = dt;
    config["gravity"]                 = gravity;
    config["contact"]["enable"]       = stack;
    config["contact"]["d_hat"]        = Float(0.01);
    config["newton"]["max_iter"]      = 100;
    config["newton"]["velocity_tol"]  = Float(0.01);
    config["newton"]["transrate_tol"] = Float(0.01);
    config["contact"]["adaptive"]["min_kappa"]  = Float(1e8);
    config["contact"]["adaptive"]["init_kappa"] = Float(1e8);
    config["contact"]["adaptive"]["max_kappa"]  = Float(1e8);
    Scene scene{config};
    scene.contact_tabular().default_model(Float(0.5), Float(1e8));
    auto contact = scene.contact_tabular().default_element();
    vector<S<GeometrySlotT<SimplicialComplex>>> slots;
    for(int i = 0; i < (stack ? 5 : 1); ++i)
    {
        auto mesh = cube(mass,
                         stack ? Vector3{0, Float(0.2 + 0.4 * i), 0} :
                                 Vector3::Zero().eval());
        contact.apply_to(mesh);
        if(name == "force")
        {
            Vector12 force = Vector12::Zero();
            force[0] = mass;  // 1 m/s^2, including a small PCG right-hand side.
            AffineBodyExternalBodyForce{}.apply_to(mesh, force);
            view(*mesh.instances().find<IndexT>(builtin::is_constrained))[0] = 1;
        }
        auto object       = scene.objects().create("box-" + std::to_string(i));
        auto [slot, rest] = object->geometries().create(mesh);
        slots.push_back(slot);
    }
    if(stack)
    {
        auto plane = ground(Float(0));
        contact.apply_to(plane);
        scene.objects().create("ground")->geometries().create(plane);
    }
    world.init(scene);
    require(world.is_valid(), "GPU scene initialization failed");
    const Vector3 velocity{1, Float(-0.5), Float(0.25)};
    if(name == "initial-velocity")
    {
        auto accessor = world.features().find<AffineBodyStateAccessorFeature>();
        require(accessor != nullptr, "Missing GPU affine-body state accessor");
        auto state  = accessor->create_geometry();
        auto values = state.instances().create<Matrix4x4>(builtin::velocity,
                                                          Matrix4x4::Zero());
        view(*values)[0].block<3, 1>(0, 3) = velocity;
        accessor->copy_from(state);
    }
    vector<Vector3> previous(slots.size(), Vector3::Zero());
    for(int step = 1; step <= (stack ? 200 : 10); ++step)
    {
        advance(world);
        Float last_y = 0;
        for(std::size_t i = 0; i < slots.size(); ++i)
        {
            const Matrix4x4 pose = slots[i]->geometry().transforms().view()[0];
            require(pose.allFinite(), "Non-finite body transform");
            const Vector3 position = pose.block<3, 1>(0, 3);
            if(stack)
            {
                const Float extent =
                    Float(0.15) * pose.block<1, 3>(1, 0).cwiseAbs().sum();
                require(position.y() - extent >= Float(-1e-4), "Ground penetration");
                if(step == 200)
                {
                    require(i == 0 || position.y() - last_y >= Float(0.3 - 1e-4),
                            "Stack spacing collapsed");
                    require(std::abs(position.x()) < Float(0.02)
                                && std::abs(position.z()) < Float(0.02),
                            "Stack drift");
                    require((position - previous[i]).norm() / dt < Float(0.05),
                            "Stack has not settled");
                    require((pose.block<3, 3>(0, 0).transpose() * pose.block<3, 3>(0, 0)
                             - Matrix3x3::Identity())
                                    .norm()
                                < Float(1e-4),
                            "Rigid-body shape drift");
                }
                last_y = position.y();
            }
            else
            {
                Vector3 expected =
                    gravity * dt * dt * Float(step * (step + 1)) / Float(2);
                if(name == "force")
                    expected =
                        Vector3{dt * dt * Float(step * (step + 1)) / Float(2), 0, 0};
                if(name == "initial-velocity")
                    expected = velocity * dt * Float(step);
                require((position - expected).norm()
                            <= Float(1e-5) + Float(1e-4) * expected.norm(),
                        "Analytical trajectory mismatch at step " + std::to_string(step));
            }
            previous[i] = position;
        }
    }
}
}  // namespace

int main(int argc, char** argv)
{
    try
    {
        require(argc == 3, "Usage: dlan_physics <free-fall|force|initial-velocity|stack> <mass>");
        const std::string name = argv[1];
        require(name == "free-fall" || name == "force"
                    || name == "initial-velocity" || name == "stack",
                "Unknown physics case");
        const Float mass = std::stof(argv[2]);
        require(std::isfinite(mass) && mass > 0, "Invalid mass");
        auto config = uipc::default_config();
        config["module_dir"] = (std::filesystem::current_path() / "Release/bin").string();
        uipc::init(config);
        run(name, mass);
        std::cout << "DLAN FP32 physics passed: " << name << " mass=" << mass << '\n';
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
