#include <catch2/catch_test_macros.hpp>

#include <memory>

#include <jetstream/module_interface.hh>
#include <jetstream/registry.hh>

#include <stelline/nexus_bridge/module.hh>

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
