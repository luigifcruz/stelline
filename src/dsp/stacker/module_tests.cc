#include <algorithm>
#include <any>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <stelline/config.hh>
#include <stelline/stacker/module.hh>
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

TEST_CASE("Stacker module exposes selected implementations",
          "[stelline][stacker][module][registration]") {
    const auto implementations = Registry::ListAvailableModules("stacker");
    std::size_t expected = 0;

#if defined(STELLINE_DEVICE_CPU_AVAILABLE)
    REQUIRE(HasImplementation("stacker", DeviceType::CPU));
    ++expected;
#else
    REQUIRE_FALSE(HasImplementation("stacker", DeviceType::CPU));
#endif

#if defined(STELLINE_DEVICE_CUDA_AVAILABLE)
    REQUIRE(HasImplementation("stacker", DeviceType::CUDA));
    ++expected;
#else
    REQUIRE_FALSE(HasImplementation("stacker", DeviceType::CUDA));
#endif

    REQUIRE(implementations.size() == expected);
    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                                    << " Runtime: " << implementation.runtime) {
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("stacker",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);
            REQUIRE(module != nullptr);
        }
    }
}

TEST_CASE("Stacker CPU fills slots only from one timed RF window",
          "[stelline][stacker][module][cpu][timing]") {
#if !defined(STELLINE_DEVICE_CPU_AVAILABLE)
    SKIP("Stelline was not built with CPU support.");
#else
    constexpr U64 kBlockDuration = 2;
    for (const U64 ratio : {1, 2}) {
        for (const U64 threshold : {1, 64}) {
            DYNAMIC_SECTION("Ratio: " << ratio << ", copy threshold: " << threshold) {
                std::shared_ptr<Module> module;
                REQUIRE(Registry::BuildModule("stacker",
                                              DeviceType::CPU,
                                              RuntimeType::NATIVE,
                                              "generic",
                                              module) == Result::SUCCESS);

                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {1, 1, 2, 1}) ==
                        Result::SUCCESS);
                TensorMap inputs;
                inputs["buffer"].requested("source", "buffer");
                inputs["buffer"].tensor = input;

                Modules::Stacker config;
                config.axis = 2;
                config.ratio = ratio;
                config.copySizeThreshold = threshold;
                REQUIRE(module->create("stacker", config, inputs) == Result::SUCCESS);
                auto* runtime = module->getImpl<NativeCpuRuntimeContext>();
                REQUIRE(runtime != nullptr);
                REQUIRE(runtime->computeInitialize() == Result::SUCCESS);

                const auto submit = [&](const U64 block) {
                    REQUIRE(input.setAttribute("timestamp", block * kBlockDuration) ==
                            Result::SUCCESS);
                    REQUIRE(input.setAttribute("duration", kBlockDuration) == Result::SUCCESS);
                    std::fill_n(input.data<CF32>(), input.size(),
                                CF32{static_cast<F32>(block), 0.0f});
                    return runtime->computeSubmit();
                };

                if (ratio == 1) {
                    REQUIRE(submit(1) == Result::SUCCESS);
                    REQUIRE(submit(1) == Result::SKIP);
                    REQUIRE(submit(0) == Result::SKIP);
                } else {
                    for (const U64 block : {1, 2, 4, 6, 4, 6}) {
                        REQUIRE(submit(block) == Result::SKIP);
                    }
                }
                REQUIRE(submit(7) == Result::SUCCESS);

                const Tensor output = module->outputs().at("buffer").tensor;
                const std::vector<CF32> expected = ratio == 1
                    ? std::vector<CF32>{7.0f, 7.0f}
                    : std::vector<CF32>{6.0f, 6.0f, 7.0f, 7.0f};
                REQUIRE(std::vector<CF32>(output.data<CF32>(),
                                          output.data<CF32>() + output.size()) == expected);
                const U64 firstBlock = ratio == 1 ? 7 : 6;
                REQUIRE(std::any_cast<U64>(output.attribute("timestamp")) ==
                        firstBlock * kBlockDuration);
                REQUIRE(std::any_cast<U64>(output.attribute("duration")) ==
                        ratio * kBlockDuration);

                REQUIRE(runtime->computeDeinitialize() == Result::SUCCESS);
                REQUIRE(module->destroy() == Result::SUCCESS);
            }
        }
    }
#endif
}
