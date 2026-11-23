#include <algorithm>
#include <any>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <stelline/config.hh>
#include <stelline/correlator/module.hh>
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

TEST_CASE("Correlator module exposes selected implementations",
          "[stelline][correlator][module][registration]") {
    const auto implementations = Registry::ListAvailableModules("stelline_correlator");
    std::size_t expected = 0;

#if defined(STELLINE_DEVICE_CPU_AVAILABLE)
    REQUIRE(HasImplementation("stelline_correlator", DeviceType::CPU));
    ++expected;
#else
    REQUIRE_FALSE(HasImplementation("stelline_correlator", DeviceType::CPU));
#endif

#if defined(STELLINE_DEVICE_CUDA_AVAILABLE)
    REQUIRE(HasImplementation("stelline_correlator", DeviceType::CUDA));
    ++expected;
#else
    REQUIRE_FALSE(HasImplementation("stelline_correlator", DeviceType::CUDA));
#endif

    REQUIRE(implementations.size() == expected);
    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                                    << " Runtime: " << implementation.runtime) {
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("stelline_correlator",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);
            REQUIRE(module != nullptr);
        }
    }
}

TEST_CASE("Correlator CPU integrates only blocks of one timed RF window",
          "[stelline][correlator][module][cpu][timing]") {
#if !defined(STELLINE_DEVICE_CPU_AVAILABLE)
    SKIP("Stelline was not built with CPU support.");
#else
    constexpr U64 kBlockDuration = 8;
    for (const U64 rate : {1, 2}) {
        for (const DataType dtype : {DataType::CF32, DataType::CI8}) {
            DYNAMIC_SECTION("Rate: " << rate << ", dtype: " << dtype) {
                std::shared_ptr<Module> module;
                REQUIRE(Registry::BuildModule("stelline_correlator",
                                              DeviceType::CPU,
                                              RuntimeType::NATIVE,
                                              "generic",
                                              module) == Result::SUCCESS);

                const U64 samples = dtype == DataType::CI8 ? 4 : 1;
                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, dtype, {1, 1, samples, 2}) ==
                        Result::SUCCESS);
                TensorMap inputs;
                inputs["buffer"].requested("source", "buffer");
                inputs["buffer"].tensor = input;

                Modules::Correlator config;
                config.integrationRate = rate;
                config.calculationMode = "single_precision_fp";
                REQUIRE(module->create("correlator", config, inputs) == Result::SUCCESS);
                auto* runtime = module->getImpl<NativeCpuRuntimeContext>();
                REQUIRE(runtime != nullptr);
                REQUIRE(runtime->computeInitialize() == Result::SUCCESS);

                const auto submit = [&](const U64 block) {
                    REQUIRE(input.setAttribute("timestamp", block * kBlockDuration) ==
                            Result::SUCCESS);
                    REQUIRE(input.setAttribute("duration", kBlockDuration) == Result::SUCCESS);
                    if (dtype == DataType::CI8) {
                        std::fill_n(input.data<CI8>(), input.size(),
                                    CI8{static_cast<I8>(block), 0});
                    } else {
                        std::fill_n(input.data<CF32>(), input.size(),
                                    CF32{static_cast<F32>(block), 0.0f});
                    }
                    return runtime->computeSubmit();
                };

                F32 power = 0.0f;
                if (rate == 1) {
                    REQUIRE(submit(3) == Result::SUCCESS);
                    REQUIRE(submit(3) == Result::SKIP);
                    REQUIRE(submit(2) == Result::SKIP);
                    power = 7.0f * 7.0f;
                } else {
                    for (const U64 block : {3, 4, 6, 4, 6}) {
                        REQUIRE(submit(block) == Result::SKIP);
                    }
                    power = 6.0f * 6.0f + 7.0f * 7.0f;
                }
                REQUIRE(submit(7) == Result::SUCCESS);

                const Tensor output = module->outputs().at("buffer").tensor;
                const CF32 expected = {static_cast<F32>(samples) * power, 0.0f};
                REQUIRE(std::vector<CF32>(output.data<CF32>(),
                                          output.data<CF32>() + output.size()) ==
                        std::vector<CF32>(4, expected));
                const U64 firstBlock = rate == 1 ? 7 : 6;
                REQUIRE(std::any_cast<U64>(output.attribute("timestamp")) ==
                        firstBlock * kBlockDuration);
                REQUIRE(std::any_cast<U64>(output.attribute("duration")) ==
                        rate * kBlockDuration);

                REQUIRE(runtime->computeDeinitialize() == Result::SUCCESS);
                REQUIRE(module->destroy() == Result::SUCCESS);
            }
        }
    }
#endif
}
