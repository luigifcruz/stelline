#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <utility>

#include <jetstream/memory/axis.hh>
#include <jetstream/module_interface.hh>
#include <jetstream/registry.hh>

#include <stelline/fbh5_writer/module.hh>

using namespace Jetstream;

namespace {

std::shared_ptr<Module> BuildModule() {
    const auto implementations = Registry::ListAvailableModules("fbh5_writer");
    REQUIRE(implementations.size() == 1);

    const auto& implementation = implementations.front();
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("fbh5_writer",
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

Tensor WriterInput(const DataType dtype, const Shape& shape) {
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, dtype, shape) == Result::SUCCESS);
    REQUIRE(SetSignalAxes(input, {
        .sample = Index{0},
        .channel = Index{2},
    }) == Result::SUCCESS);
    return input;
}

}  // namespace

TEST_CASE("FBH5 writer validates semantics before defining its interface",
          "[stelline][fbh5_writer][module][validation][lifecycle]") {
    auto module = BuildModule();
    auto input = WriterInput(DataType::F32, {1, 2, 3});

    Modules::Fbh5Writer config;
    REQUIRE(module->create("test", config, InputMap(std::move(input))) ==
            Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->inputs().empty());
    REQUIRE(module->outputs().empty());
}

TEST_CASE("FBH5 writer delegates missing inputs to the framework",
          "[stelline][fbh5_writer][module][validation][inputs]") {
    auto module = BuildModule();

    Modules::Fbh5Writer config;
    REQUIRE(module->create("test", config, {}) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->inputs() ==
            Module::Interface::EntryList{"input"});
}

TEST_CASE("FBH5 writer provider rejects unsupported dtypes during validation",
          "[stelline][fbh5_writer][module][validation][dtype]") {
    auto module = BuildModule();
    auto input = WriterInput(DataType::F64, {1, 2, 3, 1});

    Modules::Fbh5Writer config;
    REQUIRE(module->create("test", config, InputMap(std::move(input))) ==
            Result::ERROR);
    REQUIRE(module->interface()->inputs().empty());
}
