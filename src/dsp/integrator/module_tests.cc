#include <algorithm>
#include <any>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <stelline/config.hh>
#include <stelline/integrator/module.hh>
#include <jetstream/module.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cpu.hh>

#include "../block_window.hh"
#include "module_impl.hh"

using namespace Jetstream;
using Modules::BlockTiming;
using Modules::BlockWindow;
using Modules::WindowStep;

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

void SetAttribute(Tensor& tensor, const std::string& key, const std::any& value) {
    if (value.has_value()) {
        REQUIRE(tensor.setAttribute(key, value) == Result::SUCCESS);
    } else {
        REQUIRE(tensor.removeAttribute(key) == Result::SUCCESS);
    }
}

WindowStep Admit(BlockWindow& window, const std::any& timestamp, const std::any& duration) {
    Tensor block;
    REQUIRE(block.create(DeviceType::CPU, DataType::CF32, {1}) == Result::SUCCESS);
    SetAttribute(block, "timestamp", timestamp);
    SetAttribute(block, "duration", duration);
    return window.admit(block);
}

WindowStep Admit(BlockWindow& window, const BlockTiming timing) {
    return Admit(window, timing.timestamp, timing.duration);
}

constexpr U64 kBlockDuration = 2;

WindowStep AdmitBlock(BlockWindow& window, const U64 block) {
    return Admit(window, BlockTiming{block * kBlockDuration, kBlockDuration});
}

std::optional<BlockTiming> Stamped(const BlockWindow& window) {
    Tensor output;
    REQUIRE(output.create(DeviceType::CPU, DataType::CF32, {1}) == Result::SUCCESS);
    REQUIRE(output.setAttribute("timestamp", U64{1}) == Result::SUCCESS);
    REQUIRE(output.setAttribute("duration", U64{1}) == Result::SUCCESS);
    REQUIRE(window.stamp(output) == Result::SUCCESS);
    if (!output.hasAttribute("timestamp")) {
        REQUIRE_FALSE(output.hasAttribute("duration"));
        return std::nullopt;
    }
    const auto timing = Modules::ReadBlockTiming(output);
    REQUIRE(timing.has_value());
    return timing;
}

void RequireStamp(const BlockWindow& window, const U64 timestamp, const U64 duration) {
    const auto timing = Stamped(window);
    REQUIRE(timing.has_value());
    REQUIRE(timing->timestamp == timestamp);
    REQUIRE(timing->duration == duration);
}

void RequireRejected(BlockWindow& window, std::initializer_list<U64> blocks) {
    for (const U64 block : blocks) {
        INFO("Block: " << block);
        REQUIRE_FALSE(AdmitBlock(window, block).accept);
    }
}

void RequireWindow(BlockWindow& window, const U64 firstBlock, const U64 blocks) {
    for (U64 index = 0; index < blocks; ++index) {
        INFO("Block: " << firstBlock + index);
        const WindowStep step = AdmitBlock(window, firstBlock + index);
        REQUIRE(step.accept);
        REQUIRE(step.index == index);
        REQUIRE(step.first == (index == 0));
        REQUIRE(step.last == (index + 1 == blocks));
    }
    RequireStamp(window, firstBlock * kBlockDuration, blocks * kBlockDuration);
}

}  // namespace

TEST_CASE("Integrator module exposes selected implementations",
          "[stelline][integrator][module][registration]") {
    const auto implementations = Registry::ListAvailableModules("stelline_integrator");
    std::size_t expected = 0;

#if defined(STELLINE_DEVICE_CPU_AVAILABLE)
    REQUIRE(HasImplementation("stelline_integrator", DeviceType::CPU));
    ++expected;
#else
    REQUIRE_FALSE(HasImplementation("stelline_integrator", DeviceType::CPU));
#endif

#if defined(STELLINE_DEVICE_CUDA_AVAILABLE)
    REQUIRE(HasImplementation("stelline_integrator", DeviceType::CUDA));
    ++expected;
#else
    REQUIRE_FALSE(HasImplementation("stelline_integrator", DeviceType::CUDA));
#endif

    REQUIRE(implementations.size() == expected);
    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                                    << " Runtime: " << implementation.runtime) {
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("stelline_integrator",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);
            REQUIRE(module != nullptr);
        }
    }
}

TEST_CASE("Integrator module reconfiguration is transactional",
          "[stelline][integrator][module][reconfigure]") {
#if !defined(STELLINE_DEVICE_CPU_AVAILABLE)
    SKIP("Stelline was not built with CPU support.");
#else
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("stelline_integrator",
                                  DeviceType::CPU,
                                  RuntimeType::NATIVE,
                                  "generic",
                                  module) == Result::SUCCESS);

    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {1, 1, 4, 1}) ==
            Result::SUCCESS);
    TensorMap inputs;
    inputs["buffer"].requested("source", "buffer");
    inputs["buffer"].tensor = input;

    Modules::Integrator config;
    REQUIRE(module->create("integrator", config, inputs) == Result::SUCCESS);
    REQUIRE(module->state() == Module::State::CREATED);

    Parser::Map invalid;
    invalid["size"] = U64{0};
    REQUIRE(module->reconfigure(invalid) == Result::ERROR);
    REQUIRE(module->state() == Module::State::CREATED);

    Parser::Map saved;
    REQUIRE(module->config(saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<U64>(saved.at("size")) == config.size);
    REQUIRE(module->destroy() == Result::SUCCESS);
#endif
}

TEST_CASE("Block windows start on the RF grid and reject stale or misaligned blocks",
          "[stelline][integrator][timing][window]") {
    constexpr U64 max = std::numeric_limits<U64>::max();
    BlockWindow window;
    window.reset(4);

    SECTION("aligned start emits the first window") {
        RequireWindow(window, 0, 4);
    }

    SECTION("late start waits for the next grid boundary") {
        RequireRejected(window, {1, 2, 3});
        RequireWindow(window, 4, 4);
    }

    SECTION("a gap abandons its window and recovery starts on the grid") {
        REQUIRE(AdmitBlock(window, 0).accept);
        REQUIRE(AdmitBlock(window, 1).accept);
        RequireRejected(window, {3, 5, 6, 7});
        RequireWindow(window, 8, 4);
    }

    SECTION("an aligned block after a gap restarts the window") {
        REQUIRE(AdmitBlock(window, 0).accept);
        REQUIRE(AdmitBlock(window, 1).accept);
        RequireWindow(window, 4, 4);
    }

    SECTION("stale and duplicate blocks neither spoil nor replay a window") {
        RequireWindow(window, 0, 4);
        RequireRejected(window, {0, 1, 2, 3});
        REQUIRE(AdmitBlock(window, 4).index == 0);
        REQUIRE(AdmitBlock(window, 5).index == 1);
        RequireRejected(window, {0, 3, 5});
        REQUIRE_FALSE(Admit(window, BlockTiming{8, 1}).accept);
        REQUIRE(AdmitBlock(window, 6).index == 2);
        REQUIRE(AdmitBlock(window, 7).last);
        RequireStamp(window, 8, 8);
    }

    SECTION("only exact boundaries qualify, with overflow checked at both ends") {
        REQUIRE_FALSE(Admit(window, BlockTiming{0, max / 4 + 1}).accept);
        REQUIRE_FALSE(Admit(window, BlockTiming{max - 2, 3}).accept);
        for (const U64 offset : {7, 5, 3, 1}) {
            REQUIRE_FALSE(Admit(window, BlockTiming{max - offset, 2}).accept);
        }
        for (const U64 timestamp : {1, 3, 5, 7}) {
            REQUIRE_FALSE(Admit(window, BlockTiming{timestamp, 2}).accept);
        }
        RequireWindow(window, 4, 4);

        const U64 lastValidStart = max - 15;
        for (const U64 offset : {0, 2, 4}) {
            const WindowStep step = Admit(window, BlockTiming{lastValidStart + offset, 2});
            REQUIRE(step.accept);
            REQUIRE_FALSE(step.last);
        }
        REQUIRE(Admit(window, BlockTiming{lastValidStart + 6, 2}).last);
        RequireStamp(window, lastValidStart, 8);
    }

    SECTION("a span change restarts on its own grid without mixing spans") {
        REQUIRE(AdmitBlock(window, 0).accept);
        REQUIRE_FALSE(Admit(window, BlockTiming{2, 3}).accept);
        for (const U64 timestamp : {12, 15, 18}) {
            REQUIRE_FALSE(Admit(window, BlockTiming{timestamp, 3}).last);
        }
        REQUIRE(Admit(window, BlockTiming{21, 3}).last);
        RequireStamp(window, 12, 12);
    }

    SECTION("partial, mistyped, zero and overflowing timing are rejected, not counted") {
        REQUIRE(AdmitBlock(window, 0).first);
        REQUIRE_FALSE(Admit(window, U64{2}, {}).accept);
        REQUIRE_FALSE(Admit(window, {}, U64{2}).accept);
        REQUIRE_FALSE(Admit(window, I64{2}, U64{2}).accept);
        REQUIRE_FALSE(Admit(window, U64{2}, I64{2}).accept);
        REQUIRE_FALSE(Admit(window, U64{2}, U64{0}).accept);
        REQUIRE_FALSE(Admit(window, max, U64{1}).accept);
        REQUIRE(AdmitBlock(window, 1).index == 1);
        REQUIRE(AdmitBlock(window, 2).index == 2);
        REQUIRE(AdmitBlock(window, 3).last);
        RequireStamp(window, 0, 8);
    }

    SECTION("untimed input counts blocks and clears the output timing") {
        for (U64 index = 0; index < 4; ++index) {
            const WindowStep step = Admit(window, {}, {});
            REQUIRE(step.accept);
            REQUIRE(step.index == index);
            REQUIRE(step.last == (index == 3));
        }
        REQUIRE_FALSE(Stamped(window).has_value());
    }

    SECTION("untimed input drops a partial timed window but keeps the RF watermark") {
        RequireWindow(window, 0, 4);
        REQUIRE(AdmitBlock(window, 4).accept);
        for (U64 index = 0; index < 4; ++index) {
            const WindowStep step = Admit(window, {}, {});
            REQUIRE(step.index == index);
            REQUIRE(step.last == (index == 3));
        }
        REQUIRE_FALSE(Stamped(window).has_value());
        RequireRejected(window, {0, 1, 2, 3, 4});
        RequireWindow(window, 8, 4);
    }

    SECTION("single-block windows keep unaligned timing and still reject stale blocks") {
        window.reset(1);
        const WindowStep step = Admit(window, BlockTiming{3, 8});
        REQUIRE(step.accept);
        REQUIRE(step.first);
        REQUIRE(step.last);
        RequireStamp(window, 3, 8);
        REQUIRE_FALSE(Admit(window, BlockTiming{3, 8}).accept);
        REQUIRE_FALSE(Admit(window, BlockTiming{5, 8}).accept);
        REQUIRE(Admit(window, BlockTiming{11, 8}).last);
        RequireStamp(window, 11, 8);
    }
}

TEST_CASE("Integrator CPU sums only blocks of one timed RF window",
          "[stelline][integrator][module][cpu][timing]") {
#if !defined(STELLINE_DEVICE_CPU_AVAILABLE)
    SKIP("Stelline was not built with CPU support.");
#else
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("stelline_integrator",
                                  DeviceType::CPU,
                                  RuntimeType::NATIVE,
                                  "generic",
                                  module) == Result::SUCCESS);

    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {1, 1, 4, 1}) ==
            Result::SUCCESS);
    TensorMap inputs;
    inputs["buffer"].requested("source", "buffer");
    inputs["buffer"].tensor = input;

    Modules::Integrator config;
    config.rate = 2;
    config.size = 2;
    REQUIRE(module->create("integrator", config, inputs) == Result::SUCCESS);
    auto* runtime = module->getImpl<NativeCpuRuntimeContext>();
    REQUIRE(runtime != nullptr);
    REQUIRE(runtime->computeInitialize() == Result::SUCCESS);

    const auto submit = [&](const U64 timestamp, const std::vector<CF32>& samples) {
        REQUIRE(input.setAttribute("timestamp", timestamp) == Result::SUCCESS);
        REQUIRE(input.setAttribute("duration", U64{8}) == Result::SUCCESS);
        std::copy(samples.begin(), samples.end(), input.data<CF32>());
        return runtime->computeSubmit();
    };
    const Tensor output = module->outputs().at("buffer").tensor;
    const auto requireOutput = [&](const std::vector<CF32>& expected, const U64 timestamp) {
        REQUIRE(std::vector<CF32>(output.data<CF32>(), output.data<CF32>() + output.size()) ==
                expected);
        REQUIRE(std::any_cast<U64>(output.attribute("timestamp")) == timestamp);
        REQUIRE(std::any_cast<U64>(output.attribute("duration")) == 16);
    };
    const std::vector<CF32> poison(4, CF32{99.0f, 0.0f});

    REQUIRE(submit(8, poison) == Result::SKIP);
    REQUIRE(submit(16, {1.0f, 2.0f, 4.0f, 8.0f}) == Result::SKIP);
    REQUIRE(submit(24, {16.0f, 32.0f, 64.0f, 128.0f}) == Result::SUCCESS);
    requireOutput({51.0f, 204.0f}, 16);

    REQUIRE(submit(24, poison) == Result::SKIP);
    REQUIRE(submit(32, poison) == Result::SKIP);
    REQUIRE(submit(48, {1.0f, 1.0f, 1.0f, 1.0f}) == Result::SKIP);
    REQUIRE(submit(56, {2.0f, 2.0f, 2.0f, 2.0f}) == Result::SUCCESS);
    requireOutput({6.0f, 6.0f}, 48);

    REQUIRE(runtime->computeDeinitialize() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
#endif
}

TEST_CASE("Integrator CPU bypass passes each new timed block through",
          "[stelline][integrator][module][cpu][timing]") {
#if !defined(STELLINE_DEVICE_CPU_AVAILABLE)
    SKIP("Stelline was not built with CPU support.");
#else
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("stelline_integrator",
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

    Modules::Integrator config;
    REQUIRE(module->create("integrator", config, inputs) == Result::SUCCESS);
    auto* runtime = module->getImpl<NativeCpuRuntimeContext>();
    REQUIRE(runtime != nullptr);
    REQUIRE(runtime->computeInitialize() == Result::SUCCESS);

    REQUIRE(input.setAttribute("timestamp", U64{3}) == Result::SUCCESS);
    REQUIRE(runtime->computeSubmit() == Result::SKIP);
    REQUIRE(input.setAttribute("duration", U64{8}) == Result::SUCCESS);
    REQUIRE(runtime->computeSubmit() == Result::SUCCESS);
    REQUIRE(runtime->computeSubmit() == Result::SKIP);
    REQUIRE(input.setAttribute("timestamp", U64{11}) == Result::SUCCESS);
    REQUIRE(runtime->computeSubmit() == Result::SUCCESS);
    const Tensor output = module->outputs().at("buffer").tensor;
    REQUIRE(std::any_cast<U64>(output.attribute("timestamp")) == 11);
    REQUIRE(std::any_cast<U64>(output.attribute("duration")) == 8);

    REQUIRE(runtime->computeDeinitialize() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
#endif
}
