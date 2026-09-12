#include <catch2/catch_test_macros.hpp>

#include <memory>

#include <jetstream/module_interface.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_python.hh>

#include <stelline/nexus_bridge/module.hh>

#include "module_impl.hh"

using namespace Jetstream;

namespace {

std::shared_ptr<Module> BuildModule() {
    const auto implementations = Registry::ListAvailableModules("nexus_bridge");
    REQUIRE(implementations.size() == 1);

    const auto& implementation = implementations.front();
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("nexus_bridge",
                                  implementation.device,
                                  implementation.runtime,
                                  implementation.provider,
                                  module) == Result::SUCCESS);
    return module;
}

}  // namespace

TEST_CASE("Nexus bridge rejects an empty URL before creating Python",
          "[stelline][nexus_bridge][module][validation][lifecycle]") {
    auto module = BuildModule();

    Modules::NexusBridge config;
    config.url.clear();
    REQUIRE(module->create("test", config, {}) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->inputs().empty());
    REQUIRE(module->interface()->outputs().empty());
    REQUIRE(module->outputs().empty());
    REQUIRE(module->taint() == Module::Taint::CLEAN);
}

TEST_CASE("Nexus bridge reloads its Python source after runtime unloading",
          "[stelline][nexus_bridge][module][python][lifecycle]") {
    const auto validation = PythonRuntimeContext::ValidateRuntimePath("");
    if (!validation.valid) {
        SKIP("Optional Python runtime is unavailable: " << validation.message);
    }

    PythonRuntimeContext offlineConvex;
    REQUIRE(offlineConvex.createCompute(R"PY(
import sys
import types

_saved_modules = {name: sys.modules.get(name) for name in ("convex", "convex.values")}

class ConvexClient:
    def __init__(self, url):
        raise RuntimeError("Offline lifecycle test")

convex = types.ModuleType("convex")
convex.ConvexClient = ConvexClient
values = types.ModuleType("convex.values")
values.ConvexInt64 = type("ConvexInt64", (), {})
sys.modules["convex"] = convex
sys.modules["convex.values"] = values

def compute(ctx):
    pass

def cleanup():
    for name, module in _saved_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
)PY", {}, {}, {}, {}, {}) == Result::SUCCESS);

    auto module = BuildModule();
    Modules::NexusBridge config;
    config.url = "https://nexus.invalid";
    REQUIRE(module->create("nexus_reload", config, {}) == Result::SUCCESS);

    auto* runtime = module->getImpl<PythonRuntimeContext>();
    REQUIRE(runtime != nullptr);
    REQUIRE(runtime->diagnostic().healthy);
    REQUIRE(runtime->unloadCompute() == Result::SUCCESS);
    REQUIRE(runtime->loadCompute() == Result::SUCCESS);
    REQUIRE(runtime->diagnostic().healthy);

    REQUIRE(module->destroy() == Result::SUCCESS);
    REQUIRE(offlineConvex.destroyCompute() == Result::SUCCESS);
}
