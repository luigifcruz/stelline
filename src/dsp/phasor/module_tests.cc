#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <stelline/config.hh>
#include <stelline/phasor/module.hh>
#include <jetstream/module.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cpu.hh>

#include "module_impl.hh"

using namespace Jetstream;

namespace {

bool HasImplementation(const std::string& type, const DeviceType device) {
    const auto implementations = Registry::ListAvailableModules(type);
    return std::any_of(implementations.begin(), implementations.end(),
                       [device](const auto& implementation) {
                           return implementation.device == device &&
                                  implementation.runtime == RuntimeType::NATIVE &&
                                  implementation.provider == "generic";
                       });
}

}  // namespace

TEST_CASE("Phasor module exposes its selected CPU implementation",
          "[stelline][phasor][module][registration]") {
    const auto implementations = Registry::ListAvailableModules("phasor");
    std::size_t expected = 0;

#if defined(STELLINE_DEVICE_CPU_AVAILABLE)
    REQUIRE(HasImplementation("phasor", DeviceType::CPU));
    ++expected;
#else
    REQUIRE_FALSE(HasImplementation("phasor", DeviceType::CPU));
#endif

    REQUIRE_FALSE(HasImplementation("phasor", DeviceType::CUDA));
    REQUIRE(implementations.size() == expected);

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                                    << " Runtime: " << implementation.runtime) {
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("phasor",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);
            REQUIRE(module != nullptr);
        }
    }
}

TEST_CASE("Phasor preserves calibrations for a beam at boresight",
          "[stelline][phasor][module][cpu][compute]") {
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("phasor",
                                  DeviceType::CPU,
                                  RuntimeType::NATIVE,
                                  "generic",
                                  module) == Result::SUCCESS);

    Tensor positions;
    Tensor calibrations;
    Tensor boresight;
    Tensor beams;
    Tensor julianDate;
    Tensor dut1;
    REQUIRE(positions.create(DeviceType::CPU, DataType::F64, {2, 3}) ==
            Result::SUCCESS);
    REQUIRE(calibrations.create(DeviceType::CPU, DataType::CF64, {2, 2, 2}) ==
            Result::SUCCESS);
    REQUIRE(boresight.create(DeviceType::CPU, DataType::F64, {2}) ==
            Result::SUCCESS);
    REQUIRE(beams.create(DeviceType::CPU, DataType::F64, {1, 2}) ==
            Result::SUCCESS);
    REQUIRE(julianDate.create(DeviceType::CPU, DataType::F64, {1}) ==
            Result::SUCCESS);
    REQUIRE(dut1.create(DeviceType::CPU, DataType::F64, {1}) ==
            Result::SUCCESS);

    const F64 antennaPositions[] = {6378137.0, 0.0, 0.0,
                                   6378137.0, 32.0, 4.0};
    std::copy(std::begin(antennaPositions), std::end(antennaPositions),
              positions.data<F64>());
    boresight.data<F64>()[0] = beams.data<F64>()[0] = 1.0;
    boresight.data<F64>()[1] = beams.data<F64>()[1] = 0.5;
    julianDate.data<F64>()[0] = 2451545.0;
    dut1.data<F64>()[0] = 0.0;
    for (U64 i = 0; i < calibrations.size(); ++i) {
        calibrations.data<CF64>()[i] = CF64{0.25 * (i + 1), -0.125 * (i + 1)};
    }

    TensorMap inputs;
    const auto addInput = [&inputs](const std::string& name, const Tensor& tensor) {
        inputs[name].requested("source", name);
        inputs[name].tensor = tensor;
    };
    addInput("antennaPositions", positions);
    addInput("antennaCalibrations", calibrations);
    addInput("boresightCoordinates", boresight);
    addInput("beamCoordinates", beams);
    addInput("julianDate", julianDate);
    addInput("dut1", dut1);

    Modules::Phasor config;
    config.referenceAntennaIndex = 1;
    config.arrayReferenceLongitude = 0.0;
    config.arrayReferenceLatitude = 0.0;
    config.arrayReferenceAltitude = 0.0;
    REQUIRE(module->create("phasor", config, inputs) == Result::SUCCESS);

    auto* runtime = module->getImpl<NativeCpuRuntimeContext>();
    REQUIRE(runtime != nullptr);
    REQUIRE(runtime->computeInitialize() == Result::SUCCESS);
    REQUIRE(runtime->computeSubmit() == Result::SUCCESS);

    const Tensor delays = module->outputs().at("delays").tensor;
    const Tensor phasors = module->outputs().at("phasors").tensor;
    REQUIRE(delays.shape() == Shape{1, 2});
    REQUIRE(delays.dtype() == DataType::F64);
    REQUIRE(phasors.shape() == Shape{1, 2, 2, 2});
    REQUIRE(phasors.dtype() == DataType::CF32);
    for (U64 i = 0; i < delays.size(); ++i) {
        REQUIRE_THAT(delays.data<F64>()[i], Catch::Matchers::WithinAbs(0.0, 1e-12));
    }
    for (U64 i = 0; i < phasors.size(); ++i) {
        const CF32 expected = static_cast<CF32>(calibrations.data<CF64>()[i]);
        REQUIRE_THAT(phasors.data<CF32>()[i].real(),
                     Catch::Matchers::WithinAbs(expected.real(), 1e-6));
        REQUIRE_THAT(phasors.data<CF32>()[i].imag(),
                     Catch::Matchers::WithinAbs(expected.imag(), 1e-6));
    }

    REQUIRE(runtime->computeDeinitialize() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}
