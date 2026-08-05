#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <memory>

#include <jetstream/module_interface.hh>
#include <jetstream/registry.hh>

#include <stelline/ata_receiver/module.hh>

using namespace Jetstream;

namespace {

std::shared_ptr<Module> BuildModule() {
    const auto implementations = Registry::ListAvailableModules("ata_receiver");
    REQUIRE(implementations.size() == 1);

    const auto& implementation = implementations.front();
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("ata_receiver",
                                  implementation.device,
                                  implementation.runtime,
                                  implementation.provider,
                                  module) == Result::SUCCESS);
    return module;
}

}  // namespace

TEST_CASE("ATA receiver validates bounded plans before defining its interface",
          "[stelline][ata_receiver][module][validation][lifecycle]") {
    auto module = BuildModule();

    Modules::AtaReceiver config;
    config.interfaceAddress = "lo";
    config.workerCores = {0};
    config.subscriptions = "- 127.0.0.1:10000 -> 239.1.1.1:50000";
    config.totalBlock = {
        std::numeric_limits<U64>::max(), 96, 16, 2,
    };
    config.partialBlock = {1, 96, 16, 2};
    config.packetsPerBurst = 1;
    config.maxConcurrentBursts = 1;

    REQUIRE(module->create("test", config, {}) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->outputs().empty());
    REQUIRE(module->outputs().empty());
}
