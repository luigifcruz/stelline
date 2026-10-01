#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <limits>
#include <memory>

#include <jetstream/module_interface.hh>
#include <jetstream/registry.hh>

#include <stelline/ata_receiver/module.hh>

#include "detail/daqiri_config.hh"
#include "module_impl.hh"

using namespace Jetstream;

namespace {

struct AtaReceiverValidationImpl : Modules::AtaReceiverImpl {
    using AtaReceiverImpl::validatedPacketSizeBytes;
    using AtaReceiverImpl::validatedSubscriptions;
};

Modules::AtaReceiver ValidConfig() {
    Modules::AtaReceiver config;
    config.interfaceAddress = "lo";
    config.workerCores = {0};
    config.subscriptions = "- 127.0.0.1:10000 -> 239.1.1.1:50000";
    config.totalBlock = {1, 96, 16, 2};
    config.partialBlock = {1, 96, 16, 2};
    config.packetsPerBurst = 1;
    config.maxConcurrentBursts = 1;
    return config;
}

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

    auto config = ValidConfig();
    config.totalBlock = {
        std::numeric_limits<U64>::max(), 96, 16, 2,
    };

    REQUIRE(module->create("test", config, {}) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->outputs().empty());
    REQUIRE(module->outputs().empty());
}

TEST_CASE("ATA receiver sizes packet buffers from the fragment shape",
          "[stelline][ata_receiver][module][validation]") {
    const auto [partialBlock, expectedPayloadSize] = GENERATE(Catch::Generators::table<Shape, U64>({
        {Shape{1, 1, 512, 2}, 2048},
        {Shape{1, 96, 16, 2}, 6144},
        {Shape{1, 128, 16, 2}, 8192},
    }));
    AtaReceiverValidationImpl impl;
    auto& config = *impl.candidate();
    config = ValidConfig();
    config.partialBlock = partialBlock;
    config.totalBlock = {2, partialBlock[1] * 2, partialBlock[2] * 2, 2};
    config.dataType = GENERATE("CI8", "CF32");

    REQUIRE(impl.validate() == Result::SUCCESS);
    REQUIRE(impl.validatedPacketSizeBytes == expectedPayloadSize);

    Modules::DaqiriRxConfigParams params;
    params.interfaceAddress = config.interfaceAddress;
    params.workerCores = config.workerCores;
    params.packetsPerBurst = config.packetsPerBurst;
    params.maxConcurrentBursts = config.maxConcurrentBursts;
    params.packetSizeBytes = impl.validatedPacketSizeBytes;

    daqiri::NetworkConfig networkConfig = {};
    REQUIRE(Modules::BuildDaqiriRxConfig(params, impl.validatedSubscriptions, networkConfig) == Result::SUCCESS);
    REQUIRE(networkConfig.mrs_.at("RX_DATA_0").buf_size_ == expectedPayloadSize);
}

TEST_CASE("ATA receiver rejects overflowing fragment payloads",
          "[stelline][ata_receiver][module][validation]") {
    AtaReceiverValidationImpl impl;
    auto& config = *impl.candidate();
    config = ValidConfig();

    SECTION("Element count overflow") {
        config.partialBlock = {1, std::numeric_limits<U64>::max(), 2, 1};
    }
    SECTION("Byte count overflow") {
        config.partialBlock = {1, std::numeric_limits<U64>::max() / 2 + 1, 1, 1};
    }
    config.totalBlock = config.partialBlock;

    REQUIRE(impl.validate() == Result::ERROR);
}
