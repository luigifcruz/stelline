#include <algorithm>
#include <vector>
#include <cstddef>
#include <memory>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include <stelline/config.hh>
#include <stelline/polarizer/module.hh>
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

TEST_CASE("Polarizer module exposes selected implementations",
          "[stelline][polarizer][module][registration]") {
    const auto implementations = Registry::ListAvailableModules("polarizer");
    std::size_t expected = 0;

#if defined(STELLINE_DEVICE_CPU_AVAILABLE)
    REQUIRE(HasImplementation("polarizer", DeviceType::CPU));
    ++expected;
#else
    REQUIRE_FALSE(HasImplementation("polarizer", DeviceType::CPU));
#endif

#if defined(STELLINE_DEVICE_CUDA_AVAILABLE)
    REQUIRE(HasImplementation("polarizer", DeviceType::CUDA));
    ++expected;
#else
    REQUIRE_FALSE(HasImplementation("polarizer", DeviceType::CUDA));
#endif

    REQUIRE(implementations.size() == expected);
    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                                    << " Runtime: " << implementation.runtime) {
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("polarizer",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);
            REQUIRE(module != nullptr);
        }
    }
}

TEST_CASE("Polarizer CPU converts linear feeds to circular and single components",
          "[stelline][polarizer][module][cpu]") {
#if !defined(STELLINE_DEVICE_CPU_AVAILABLE)
    SKIP("Stelline was not built with CPU support.");
#else
    const std::vector<CF32> x = {{1.0f, 2.0f}, {-0.5f, 0.25f}};
    const std::vector<CF32> y = {{3.0f, -1.0f}, {2.0f, 4.0f}};
    const CF32 j = {0.0f, 1.0f};

    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {1, 1, 2, 2}) == Result::SUCCESS);
    for (size_t sample = 0; sample < x.size(); ++sample) {
        input.data<CF32>()[sample * 2 + 0] = x[sample];
        input.data<CF32>()[sample * 2 + 1] = y[sample];
    }
    TensorMap inputs;
    inputs["buffer"].requested("source", "buffer");
    inputs["buffer"].tensor = input;

    SECTION("Circular") {
        Modules::Polarizer config;
        config.inputPolarization = "xy";
        config.outputPolarization = "lr";
        CpuModule polarizer("polarizer", config, inputs);
        REQUIRE(polarizer.compute() == Result::SUCCESS);

        const Tensor output = polarizer.output();
        REQUIRE(output.shape() == Shape{1, 1, 2, 2});
        for (size_t sample = 0; sample < x.size(); ++sample) {
            CHECK(output.data<CF32>()[sample * 2 + 0] == x[sample] + j * y[sample]);
            CHECK(output.data<CF32>()[sample * 2 + 1] == x[sample] - j * y[sample]);
        }
    }

    for (const std::string component : {"x", "y"}) {
        DYNAMIC_SECTION("Single component " << component) {
            Modules::Polarizer config;
            config.inputPolarization = "xy";
            config.outputPolarization = component;
            CpuModule polarizer("polarizer", config, inputs);
            REQUIRE(polarizer.compute() == Result::SUCCESS);

            const Tensor output = polarizer.output();
            REQUIRE(output.shape() == Shape{1, 1, 2, 1});
            const auto& expected = component == "x" ? x : y;
            for (size_t sample = 0; sample < expected.size(); ++sample) {
                CHECK(output.data<CF32>()[sample] == expected[sample]);
            }
        }
    }
#endif
}
