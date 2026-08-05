#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <utility>

#include <jetstream/memory/axis.hh>
#include <jetstream/module_interface.hh>
#include <jetstream/registry.hh>

#include <stelline/uvh5_writer/module.hh>

using namespace Jetstream;

namespace {

std::shared_ptr<Module> BuildModule() {
    const auto implementations = Registry::ListAvailableModules("uvh5_writer");
    REQUIRE(implementations.size() == 1);

    const auto& implementation = implementations.front();
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("uvh5_writer",
                                  implementation.device,
                                  implementation.runtime,
                                  implementation.provider,
                                  module) == Result::SUCCESS);
    return module;
}

TensorMap InputMap(Tensor tensor) {
    TensorMap inputs;
    inputs["input"].requested("test", "input");
    inputs["input"].tensor = std::move(tensor);
    return inputs;
}

Tensor WriterInput(const DataType dtype) {
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, dtype, {1, 3, 4, 4}) ==
            Result::SUCCESS);
    REQUIRE(SetSignalAxes(input, {
        .sample = Index{0},
        .channel = Index{2},
    }) == Result::SUCCESS);
    return input;
}

}  // namespace

TEST_CASE("UVH5 writer validates candidates before defining its interface",
          "[stelline][uvh5_writer][module][validation][lifecycle]") {
    auto module = BuildModule();

    Modules::Uvh5Writer config;
    config.dspChannelizationRate = 0;
    REQUIRE(module->create("test", config, {}) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->inputs().empty());

    const auto& applied =
        static_cast<const Modules::Uvh5Writer&>(module->config());
    REQUIRE(applied.dspChannelizationRate == 1);
}

TEST_CASE("UVH5 writer delegates missing inputs to the framework",
          "[stelline][uvh5_writer][module][validation][inputs]") {
    auto module = BuildModule();

    Modules::Uvh5Writer config;
    REQUIRE(module->create("test", config, {}) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->inputs() ==
            Module::Interface::EntryList{"input"});
}

TEST_CASE("UVH5 writer provider rejects unsupported dtypes during validation",
          "[stelline][uvh5_writer][module][validation][dtype]") {
    auto module = BuildModule();
    auto input = WriterInput(DataType::F32);

    Modules::Uvh5Writer config;
    REQUIRE(module->create("test", config, InputMap(std::move(input))) ==
            Result::ERROR);
    REQUIRE(module->interface()->inputs().empty());
}

TEST_CASE("UVH5 writer validates timestamp metadata with its exact type",
          "[stelline][uvh5_writer][module][validation][metadata]") {
    auto module = BuildModule();
    auto input = WriterInput(DataType::CF32);
    REQUIRE(input.setAttribute("timestamp", I64{0}) == Result::SUCCESS);

    Modules::Uvh5Writer config;
    config.recording = true;
    REQUIRE(module->create("test", config, InputMap(std::move(input))) ==
            Result::ERROR);
    REQUIRE(module->interface()->inputs().empty());
    REQUIRE_FALSE(static_cast<const Modules::Uvh5Writer&>(module->config())
                      .recording);
}
