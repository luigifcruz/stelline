#include <catch2/catch_test_macros.hpp>

#include <any>
#include <limits>
#include <vector>

#include <jetstream/flowgraph.hh>
#include <jetstream/flowgraph_view.hh>
#include <jetstream/memory/axis.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>

#include <stelline/ata_receiver/block.hh>

#include "module_impl.hh"

using namespace Jetstream;

namespace {

constexpr const char* kTestProvider = "stelline-test-no-hardware";

struct AtaReceiverTestImpl final : public Modules::AtaReceiverImpl,
                                   public NativeCpuRuntimeContext,
                                   public Scheduler::Context {};

JST_REGISTER_MODULE(AtaReceiverTestImpl,
                    DeviceType::CPU,
                    RuntimeType::NATIVE,
                    kTestProvider);

Blocks::AtaReceiver ValidConfig() {
    Blocks::AtaReceiver config;
    config.interfaceAddress = "validation-only";
    config.workerCores = {0};
    config.subscriptions = "- 127.0.0.1:10000 -> 239.1.1.1:50000";
    config.totalBlock = {1, 96, 16, 2};
    config.partialBlock = {1, 96, 16, 2};
    config.packetsPerBurst = 1;
    config.maxConcurrentBursts = 1;
    config.maxConcurrentBlocks = 1;
    config.outputPoolSize = 1;
    return config;
}

}  // namespace

TEST_CASE("ATA receiver block preserves an invalid plan for sparse recovery",
          "[stelline][ata_receiver][block][reconfigure][validation]") {
    Flowgraph flowgraph;
    REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    auto invalid = ValidConfig();
    invalid.totalBlock = {
        std::numeric_limits<U64>::max(), 96, 16, 2,
    };
    REQUIRE(flowgraph.blockCreate("ata",
                                  invalid,
                                  {},
                                  DeviceType::CPU,
                                  RuntimeType::NATIVE,
                                  kTestProvider) == Result::SUCCESS);

    Flowgraph::View::BlockData block;
    REQUIRE(flowgraph.view().block("ata", block) == Result::SUCCESS);
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.outputs.empty());

    Parser::Map saved;
    REQUIRE(flowgraph.blockConfig("ata", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<std::vector<U64>>(saved.at("totalBlock")) ==
            invalid.totalBlock);

    const auto valid = ValidConfig();
    Parser::Map recovery;
    recovery["totalBlock"] = valid.totalBlock;
    REQUIRE(flowgraph.blockReconfigure("ata", recovery) == Result::SUCCESS);

    REQUIRE(flowgraph.view().block("ata", block) == Result::SUCCESS);
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.count("output") == 1);

    const Tensor output = block.outputs.at("output").tensor;
    REQUIRE(output.device() == DeviceType::CPU);
    REQUIRE(output.dtype() == DataType::CF32);
    REQUIRE(output.shape() == Shape{1, 96, 16, 2});
    REQUIRE(output.hasAttribute("timestamp"));

    SignalAxes axes;
    REQUIRE(ResolveSignalAxes(output, axes) == Result::SUCCESS);
    REQUIRE(axes.sample == Index{2});
    REQUIRE(axes.channel == Index{1});

    REQUIRE(flowgraph.blockDestroy("ata", false) == Result::SUCCESS);
    REQUIRE(flowgraph.destroy() == Result::SUCCESS);
}
