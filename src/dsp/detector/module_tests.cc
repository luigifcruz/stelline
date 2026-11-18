#include <algorithm>
#include <vector>
#include <cstddef>
#include <memory>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include <stelline/config.hh>
#include <stelline/detector/module.hh>
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

TEST_CASE("Detector module exposes selected implementations",
          "[stelline][detector][module][registration]") {
    const auto implementations = Registry::ListAvailableModules("detector");
    std::size_t expected = 0;

#if defined(STELLINE_DEVICE_CPU_AVAILABLE)
    REQUIRE(HasImplementation("detector", DeviceType::CPU));
    ++expected;
#else
    REQUIRE_FALSE(HasImplementation("detector", DeviceType::CPU));
#endif

#if defined(STELLINE_DEVICE_CUDA_AVAILABLE)
    REQUIRE(HasImplementation("detector", DeviceType::CUDA));
    ++expected;
#else
    REQUIRE_FALSE(HasImplementation("detector", DeviceType::CUDA));
#endif

    REQUIRE(implementations.size() == expected);
    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                                    << " Runtime: " << implementation.runtime) {
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("detector",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);
            REQUIRE(module != nullptr);
        }
    }
}

TEST_CASE("Detector CPU computes power products and integrates consecutive samples",
          "[stelline][detector][module][cpu]") {
#if !defined(STELLINE_DEVICE_CPU_AVAILABLE)
    SKIP("Stelline was not built with CPU support.");
#else
    const std::vector<CF32> x = {{1.0f, 2.0f}, {0.0f, 1.0f}, {-2.0f, 0.5f}, {3.0f, -1.0f}};
    const std::vector<CF32> y = {{3.0f, -1.0f}, {2.0f, 2.0f}, {1.0f, 1.0f}, {0.0f, -2.0f}};

    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {1, 1, 4, 2}) == Result::SUCCESS);
    for (size_t sample = 0; sample < x.size(); ++sample) {
        input.data<CF32>()[sample * 2 + 0] = x[sample];
        input.data<CF32>()[sample * 2 + 1] = y[sample];
    }
    TensorMap inputs;
    inputs["buffer"].requested("source", "buffer");
    inputs["buffer"].tensor = input;

    for (const U64 rate : {U64{1}, U64{2}}) {
        DYNAMIC_SECTION("Four products, rate " << rate) {
            Modules::Detector config;
            config.integrationRate = rate;
            config.numberOfOutputPolarizations = 4;
            CpuModule detector("detector", config, inputs);
            REQUIRE(detector.compute() == Result::SUCCESS);

            const Tensor output = detector.output();
            REQUIRE(output.shape() == Shape{1, 1, 4 / rate, 4});
            for (U64 group = 0; group < 4 / rate; ++group) {
                F32 xx = 0.0f, yy = 0.0f;
                CF32 xy = {};
                for (U64 index = group * rate; index < (group + 1) * rate; ++index) {
                    xx += std::norm(x[index]);
                    yy += std::norm(y[index]);
                    xy += x[index] * std::conj(y[index]);
                }
                const F32* products = output.data<F32>() + group * 4;
                CHECK(products[0] == xx);
                CHECK(products[1] == yy);
                CHECK(products[2] == xy.real());
                CHECK(products[3] == xy.imag());
            }
        }
    }

    SECTION("Total power") {
        Modules::Detector config;
        config.integrationRate = 4;
        config.numberOfOutputPolarizations = 1;
        CpuModule detector("detector", config, inputs);
        REQUIRE(detector.compute() == Result::SUCCESS);

        const Tensor output = detector.output();
        REQUIRE(output.shape() == Shape{1, 1, 1, 1});
        F32 power = 0.0f;
        for (size_t sample = 0; sample < x.size(); ++sample) {
            power += std::norm(x[sample]) + std::norm(y[sample]);
        }
        CHECK(output.data<F32>()[0] == power);
    }
#endif
}
