#include <catch2/catch_test_macros.hpp>

#include <jetstream/domains/core/ones_tensor/block.hh>
#include <jetstream/flowgraph.hh>
#include <jetstream/flowgraph_view.hh>
#include <jetstream/memory/axis.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>

#include <stelline/fbh5_writer/block.hh>

#include "module_impl.hh"

using namespace Jetstream;

namespace {

constexpr const char* kTestProvider = "stelline-test-no-hardware";

struct Fbh5WriterTestImpl final : public Modules::Fbh5WriterImpl,
                                   public NativeCpuRuntimeContext,
                                   public Scheduler::Context {};

JST_REGISTER_MODULE(Fbh5WriterTestImpl,
                    DeviceType::CPU,
                    RuntimeType::NATIVE,
                    kTestProvider);

}  // namespace

TEST_CASE("FBH5 writer block creates its inert child module",
          "[stelline][fbh5_writer][block][lifecycle]") {
    Flowgraph flowgraph;
    REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    Blocks::OnesTensor sourceConfig;
    sourceConfig.shape = {1, 2, 3, 1};
    sourceConfig.dataType = "F32";
    REQUIRE(flowgraph.blockCreate("source", sourceConfig, {}) == Result::SUCCESS);

    Flowgraph::View::BlockData source;
    REQUIRE(flowgraph.view().block("source", source) == Result::SUCCESS);
    Tensor sourceOutput = source.outputs.at("buffer").tensor;
    REQUIRE(SetSignalAxes(sourceOutput, {
        .sample = Index{0},
        .channel = Index{2},
    }) == Result::SUCCESS);

    TensorMap inputs;
    inputs["input"].requested("source", "buffer");

    Blocks::Fbh5Writer writerConfig;
    writerConfig.filepath.clear();
    writerConfig.recording = false;
    REQUIRE(flowgraph.blockCreate("writer",
                                  writerConfig,
                                  inputs,
                                  DeviceType::CPU,
                                  RuntimeType::NATIVE,
                                  kTestProvider) == Result::SUCCESS);

    Flowgraph::View::BlockData writer;
    REQUIRE(flowgraph.view().block("writer", writer) == Result::SUCCESS);
    REQUIRE(writer.state == Block::State::Created);
    REQUIRE(writer.outputs.empty());

    REQUIRE(flowgraph.blockDestroy("writer", false) == Result::SUCCESS);
    REQUIRE(flowgraph.blockDestroy("source", false) == Result::SUCCESS);
    REQUIRE(flowgraph.destroy() == Result::SUCCESS);
}
