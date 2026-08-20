#include <catch2/catch_test_macros.hpp>

#include <any>
#include <string>
#include <vector>

#include <jetstream/flowgraph.hh>
#include <jetstream/flowgraph_view.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>

#include <stelline/nexus_bridge/block.hh>

#include "module_impl.hh"

using namespace Jetstream;

namespace {

constexpr const char* kTestProvider = "stelline-test-no-hardware";

struct NexusBridgeTestImpl final : public Modules::NexusBridgeImpl,
                                   public NativeCpuRuntimeContext,
                                   public Scheduler::Context {};

JST_REGISTER_MODULE(NexusBridgeTestImpl,
                    DeviceType::CPU,
                    RuntimeType::NATIVE,
                    kTestProvider);

}  // namespace

TEST_CASE("Nexus bridge preserves an invalid URL for sparse recovery",
          "[stelline][nexus_bridge][block][reconfigure][validation]") {
    Flowgraph flowgraph;
    REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    Blocks::NexusBridge invalid;
    invalid.url.clear();
    REQUIRE(flowgraph.blockCreate("nexus",
                                  invalid,
                                  {},
                                  DeviceType::CPU,
                                  RuntimeType::NATIVE,
                                  kTestProvider) == Result::SUCCESS);

    Flowgraph::View::BlockData block;
    REQUIRE(flowgraph.view().block("nexus", block) == Result::SUCCESS);
    REQUIRE(block.state == Block::State::Errored);

    Parser::Map saved;
    REQUIRE(flowgraph.blockConfig("nexus", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<std::string>(saved.at("url")).empty());

    Parser::Map recovery;
    recovery["url"] = std::string{"https://nexus.invalid"};
    REQUIRE(flowgraph.blockReconfigure("nexus", recovery) == Result::SUCCESS);
    REQUIRE(flowgraph.view().block("nexus", block) == Result::SUCCESS);
    REQUIRE(block.state == Block::State::Created);

    std::vector<Flowgraph::View::MetricEntry> metrics;
    REQUIRE(flowgraph.view().metrics("nexus", metrics) == Result::SUCCESS);
    bool foundMetricsMonitored = false;
    for (const auto& metric : metrics) {
        if (metric.name == "metricsMonitoredDisplay") {
            foundMetricsMonitored = true;
            REQUIRE(metric.format == "label");
            REQUIRE(std::any_cast<std::string>(metric.value) == "0");
        }
    }
    REQUIRE(foundMetricsMonitored);

    REQUIRE(flowgraph.blockDestroy("nexus", false) == Result::SUCCESS);
    REQUIRE(flowgraph.destroy() == Result::SUCCESS);
}
