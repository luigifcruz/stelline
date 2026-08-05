#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <jetstream/flowgraph.hh>
#include <jetstream/flowgraph_view.hh>
#include <jetstream/memory/axis.hh>
#include <jetstream/module_interface.hh>
#include <jetstream/registry.hh>

#include <stelline/ata_receiver/module.hh>
#include <stelline/fbh5_writer/module.hh>
#include <stelline/uvh5_writer/module.hh>

using namespace Jetstream;

namespace {

std::shared_ptr<Module> BuildModule(const std::string& type) {
    const auto implementations = Registry::ListAvailableModules(type);
    REQUIRE(implementations.size() == 1);

    const auto& implementation = implementations.front();
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule(type,
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

Tensor WriterInput(const DataType dtype,
                   const Shape& shape,
                   const Index sampleAxis,
                   const Index channelAxis) {
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, dtype, shape) == Result::SUCCESS);
    REQUIRE(SetSignalAxes(input, {
        .sample = sampleAxis,
        .channel = channelAxis,
    }) == Result::SUCCESS);
    return input;
}

}  // namespace

TEST_CASE("Stelline blocks declare their child modules",
          "[stelline][blocks][requirements]") {
    for (const std::string type : {
             "ata_receiver",
             "fbh5_writer",
             "nexus_bridge",
             "uvh5_writer",
         }) {
        DYNAMIC_SECTION(type) {
            const auto registrations = Registry::ListAvailableBlocks(type);
            REQUIRE(registrations.size() == 1);
            REQUIRE(registrations.front().moduleRequirements ==
                    std::vector<Registry::BlockModuleRequirement>{{type}});
        }
    }
}

TEST_CASE("Writer semantic validation runs before interface definition",
          "[stelline][writers][validation][lifecycle]") {
    SECTION("FBH5 rank") {
        auto module = BuildModule("fbh5_writer");
        auto input = WriterInput(DataType::F32, {1, 2, 3}, 0, 2);

        Modules::Fbh5Writer config;
        REQUIRE(module->create("test", config, InputMap(std::move(input))) ==
                Result::ERROR);
        REQUIRE(module->state() == Module::State::ERRORED);
        REQUIRE(module->interface()->inputs().empty());
        REQUIRE(module->outputs().empty());
    }

    SECTION("UVH5 candidate") {
        auto module = BuildModule("uvh5_writer");
        Modules::Uvh5Writer config;
        config.dspChannelizationRate = 0;

        REQUIRE(module->create("test", config, {}) == Result::ERROR);
        REQUIRE(module->state() == Module::State::ERRORED);
        REQUIRE(module->interface()->inputs().empty());

        const auto& applied =
            static_cast<const Modules::Uvh5Writer&>(module->config());
        REQUIRE(applied.dspChannelizationRate == 1);
    }
}

TEST_CASE("Writer validation delegates missing inputs to the framework",
          "[stelline][writers][validation][inputs]") {
    for (const std::string type : {"fbh5_writer", "uvh5_writer"}) {
        DYNAMIC_SECTION(type) {
            auto module = BuildModule(type);

            if (type == "fbh5_writer") {
                Modules::Fbh5Writer config;
                REQUIRE(module->create("test", config, {}) == Result::ERROR);
            } else {
                Modules::Uvh5Writer config;
                REQUIRE(module->create("test", config, {}) == Result::ERROR);
            }

            REQUIRE(module->state() == Module::State::ERRORED);
            REQUIRE(module->interface()->inputs() ==
                    Module::Interface::EntryList{"input"});
        }
    }
}

TEST_CASE("Writer providers reject unsupported dtypes during validation",
          "[stelline][writers][validation][dtype]") {
    SECTION("FBH5") {
        auto module = BuildModule("fbh5_writer");
        auto input = WriterInput(DataType::F64, {1, 2, 3, 1}, 0, 2);

        Modules::Fbh5Writer config;
        REQUIRE(module->create("test", config, InputMap(std::move(input))) ==
                Result::ERROR);
        REQUIRE(module->interface()->inputs().empty());
    }

    SECTION("UVH5") {
        auto module = BuildModule("uvh5_writer");
        auto input = WriterInput(DataType::F32, {1, 3, 4, 4}, 0, 2);

        Modules::Uvh5Writer config;
        REQUIRE(module->create("test", config, InputMap(std::move(input))) ==
                Result::ERROR);
        REQUIRE(module->interface()->inputs().empty());
    }
}

TEST_CASE("UVH5 validates timestamp metadata with its exact type",
          "[stelline][uvh5][validation][metadata]") {
    auto module = BuildModule("uvh5_writer");
    auto input = WriterInput(DataType::CF32, {1, 3, 4, 4}, 0, 2);
    REQUIRE(input.setAttribute("timestamp", I64{0}) == Result::SUCCESS);

    Modules::Uvh5Writer config;
    config.recording = true;
    REQUIRE(module->create("test", config, InputMap(std::move(input))) ==
            Result::ERROR);
    REQUIRE(module->interface()->inputs().empty());
    REQUIRE_FALSE(static_cast<const Modules::Uvh5Writer&>(module->config())
                      .recording);
}

TEST_CASE("ATA validates bounded candidate plans before interface definition",
          "[stelline][ata][validation][lifecycle]") {
    auto module = BuildModule("ata_receiver");

    Modules::AtaReceiver config;
    config.interfaceAddress = "lo";
    config.workerCores = {0};
    config.subscriptions = "- 127.0.0.1:10000 -> 239.1.1.1:50000";
    config.totalBlock = {std::numeric_limits<U64>::max(), 96, 16, 2};
    config.partialBlock = {1, 96, 16, 2};
    config.packetsPerBurst = 1;
    config.maxConcurrentBursts = 1;

    REQUIRE(module->create("test", config, {}) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->outputs().empty());
    REQUIRE(module->outputs().empty());
}

TEST_CASE("Nexus block preserves invalid candidates for sparse recovery",
          "[stelline][nexus][block][reconfigure][validation]") {
    Flowgraph flowgraph;
    REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    Parser::Map invalid;
    invalid["url"] = std::string{};
    REQUIRE(flowgraph.blockCreate("nexus",
                                  "nexus_bridge",
                                  invalid,
                                  {},
                                  DeviceType::CPU,
                                  RuntimeType::PYTHON) == Result::SUCCESS);

    Flowgraph::View::BlockData block;
    REQUIRE(flowgraph.view().block("nexus", block) == Result::SUCCESS);
    REQUIRE(block.state == Block::State::Errored);

    Parser::Map saved;
    REQUIRE(flowgraph.blockConfig("nexus", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<std::string>(saved.at("url")).empty());

    Parser::Map recovery;
    recovery["url"] = std::string{"local-test"};
    REQUIRE(flowgraph.blockReconfigure("nexus", recovery) == Result::SUCCESS);
    REQUIRE(flowgraph.view().block("nexus", block) == Result::SUCCESS);
    REQUIRE(block.state == Block::State::Created);

    REQUIRE(flowgraph.blockDestroy("nexus", false) == Result::SUCCESS);
    REQUIRE(flowgraph.destroy() == Result::SUCCESS);
}
