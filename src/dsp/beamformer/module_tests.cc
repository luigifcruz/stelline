#include <algorithm>
#include <vector>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include <stelline/beamformer/module.hh>
#include <stelline/config.hh>
#include <jetstream/module.hh>
#include <jetstream/registry.hh>

#include "helpers.hh"

using namespace Jetstream;
using namespace Jetstream::Testing;

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

TEST_CASE("Beamformer module exposes selected implementations",
          "[stelline][beamformer][module][registration]") {
    const auto implementations = Registry::ListAvailableModules("stelline_beamformer");
    std::size_t expected = 0;

#if defined(STELLINE_DEVICE_CPU_AVAILABLE)
    REQUIRE(HasImplementation("stelline_beamformer", DeviceType::CPU));
    ++expected;
#else
    REQUIRE_FALSE(HasImplementation("stelline_beamformer", DeviceType::CPU));
#endif

#if defined(STELLINE_DEVICE_CUDA_AVAILABLE)
    REQUIRE(HasImplementation("stelline_beamformer", DeviceType::CUDA));
    ++expected;
#else
    REQUIRE_FALSE(HasImplementation("stelline_beamformer", DeviceType::CUDA));
#endif

    REQUIRE(implementations.size() == expected);
    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                                    << " Runtime: " << implementation.runtime) {
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("stelline_beamformer",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);
            REQUIRE(module != nullptr);
        }
    }
}

TEST_CASE("Beamformer CPU forms weighted beams and the incoherent beam",
          "[stelline][beamformer][module][cpu]") {
#if !defined(STELLINE_DEVICE_CPU_AVAILABLE)
    SKIP("Stelline was not built with CPU support.");
#else
    constexpr U64 kAntennas = 2;
    constexpr U64 kBeams = 2;
    const CF32 samples[kAntennas][2] = {{{1.0f, 2.0f}, {3.0f, -1.0f}}, {{0.0f, 1.0f}, {2.0f, 0.0f}}};
    const CF32 weights[kBeams][kAntennas][2] = {
        {{{0.0f, 1.5f}, {1.0f, 0.0f}}, {{0.5f, 0.0f}, {0.0f, -2.0f}}},
        {{{0.5f, 0.5f}, {-1.0f, 0.0f}}, {{2.0f, 0.0f}, {0.0f, 0.5f}}},
    };

    Tensor phasors;
    REQUIRE(phasors.create(DeviceType::CPU, DataType::CF32, {kBeams, kAntennas, 1, 1, 2}) == Result::SUCCESS);
    for (U64 beam = 0; beam < kBeams; ++beam) {
        for (U64 antenna = 0; antenna < kAntennas; ++antenna) {
            for (U64 polarization = 0; polarization < 2; ++polarization) {
                phasors.data<CF32>()[(beam * kAntennas + antenna) * 2 + polarization] =
                    weights[beam][antenna][polarization];
            }
        }
    }

    for (const DataType dtype : {DataType::CF32, DataType::CI8}) {
        for (const bool squareRoot : {false, true}) {
            DYNAMIC_SECTION("Input " << dtype << ", square root " << squareRoot) {
                const F32 scale = dtype == DataType::CI8 ? 1.0f / 128.0f : 1.0f;
                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, dtype, {kAntennas, 1, 1, 2}) == Result::SUCCESS);
                for (U64 antenna = 0; antenna < kAntennas; ++antenna) {
                    for (U64 polarization = 0; polarization < 2; ++polarization) {
                        const CF32 value = samples[antenna][polarization];
                        if (dtype == DataType::CI8) {
                            input.data<CI8>()[antenna * 2 + polarization] = {
                                static_cast<I8>(value.real()), static_cast<I8>(value.imag())};
                        } else {
                            input.data<CF32>()[antenna * 2 + polarization] = value;
                        }
                    }
                }
                TensorMap inputs;
                inputs["buffer"].requested("source", "buffer");
                inputs["buffer"].tensor = input;
                inputs["phasors"].requested("phasor", "buffer");
                inputs["phasors"].tensor = phasors;

                Modules::Beamformer config;
                config.enableIncoherentBeam = true;
                config.enableIncoherentBeamSqrt = squareRoot;
                CpuModule beamformer("stelline_beamformer", config, inputs);
                REQUIRE(beamformer.compute() == Result::SUCCESS);

                const Tensor output = beamformer.output();
                REQUIRE(output.shape() == Shape{kBeams + 1, 1, 1, 2});
                for (U64 polarization = 0; polarization < 2; ++polarization) {
                    F32 power = 0.0f;
                    for (U64 beam = 0; beam < kBeams; ++beam) {
                        CF32 expected = {};
                        for (U64 antenna = 0; antenna < kAntennas; ++antenna) {
                            expected += samples[antenna][polarization] * scale * weights[beam][antenna][polarization];
                        }
                        const CF32 actual = output.data<CF32>()[beam * 2 + polarization];
                        CHECK(std::abs(actual - expected) < 1e-6f);
                    }
                    for (U64 antenna = 0; antenna < kAntennas; ++antenna) {
                        power += std::norm(samples[antenna][polarization] * scale * weights[0][antenna][polarization]);
                    }
                    const CF32 incoherent = output.data<CF32>()[kBeams * 2 + polarization];
                    CHECK(std::abs(incoherent.real() - (squareRoot ? std::sqrt(power) : power)) < 1e-6f);
                    CHECK(incoherent.imag() == 0.0f);
                }
            }
        }
    }
#endif
}
