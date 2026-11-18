#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <memory>
#include <utility>

#include <jetstream/memory/axis.hh>
#include <jetstream/module_interface.hh>
#include <jetstream/registry.hh>

#include <stelline/uvh5_writer/module.hh>

#include "module_impl.hh"

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

TEST_CASE("UVH5 writer requires a positive RF duration before recording",
          "[stelline][uvh5_writer][module][validation][metadata]") {
    for (const bool zeroDuration : {false, true}) {
        auto module = BuildModule();
        auto input = WriterInput(DataType::CF32);
        REQUIRE(input.setAttribute("timestamp", U64{0}) == Result::SUCCESS);
        if (zeroDuration) {
            REQUIRE(input.setAttribute("duration", U64{0}) == Result::SUCCESS);
        }

        Modules::Uvh5Writer config;
        config.recording = true;
        REQUIRE(module->create("test", config, InputMap(std::move(input))) ==
                Result::ERROR);
        REQUIRE(module->state() == Module::State::ERRORED);
        REQUIRE(module->interface()->inputs().empty());
    }
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
    REQUIRE(input.setAttribute("duration", U64{16}) == Result::SUCCESS);

    Modules::Uvh5Writer config;
    config.recording = true;
    REQUIRE(module->create("test", config, InputMap(std::move(input))) ==
            Result::ERROR);
    REQUIRE(module->interface()->inputs().empty());
    REQUIRE_FALSE(static_cast<const Modules::Uvh5Writer&>(module->config())
                      .recording);
}

TEST_CASE("UVH5 marks missing integrations and clears those flags for live data",
          "[stelline][uvh5_writer][module][timing]") {
    struct TestWriter : Modules::Uvh5WriterImpl {
        using Uvh5WriterImpl::metadata;
        using Uvh5WriterImpl::prepareIntegration;
        using Uvh5WriterImpl::uvh5File;
    } writer;

    std::array<F64, 2> times{};
    std::array<F64, 2> durations{};
    std::array<F64, 8> weights{};
    std::array<H5_bool_t, 8> flags{};
    auto& file = writer.uvh5File;
    file.header.Nbls = 2;
    file.header.Nfreqs = 1;
    file.header.Npols = 4;
    file.header.time_array = times.data();
    file.header.integration_time = durations.data();
    file.nsamples = weights.data();
    file.flags = flags.data();
    writer.metadata.packetTimestampOffset = 86400;
    writer.metadata.sampleTimespan = 0.125;

    REQUIRE(writer.prepareIntegration(2, 0.25, true) == Result::SUCCESS);
    constexpr F64 kJulianDateTolerance = 2e-9;
    constexpr F64 missingTime = 2440588.5 + 0.375 / 86400.0;
    for (const F64 time : times) {
        CHECK_THAT(time, Catch::Matchers::WithinAbs(missingTime, kJulianDateTolerance));
    }
    CHECK(durations == std::array<F64, 2>{0.25, 0.25});
    for (size_t index = 0; index < weights.size(); ++index) {
        CHECK(weights[index] == 0.0);
        CHECK(flags[index] == H5_TRUE);
    }

    REQUIRE(writer.prepareIntegration(4, 0.25, false) == Result::SUCCESS);
    constexpr F64 nextTime = 2440588.5 + 0.625 / 86400.0;
    for (const F64 time : times) {
        CHECK_THAT(time, Catch::Matchers::WithinAbs(nextTime, kJulianDateTolerance));
    }
    for (size_t index = 0; index < weights.size(); ++index) {
        CHECK(weights[index] == 8.0);
        CHECK(flags[index] == H5_FALSE);
    }
}
