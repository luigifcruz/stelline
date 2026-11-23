#include <algorithm>
#include <any>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <jetstream/backend/base.hh>
#include <jetstream/flowgraph.hh>
#include <jetstream/flowgraph_view.hh>

#include <stelline/config.hh>

#include "helpers.hh"

namespace {

using namespace Jetstream;
using namespace Jetstream::Testing;

struct GraphCase {
    const char* file;
    U64 cycles;
    U64 emissionPeriod;
};

constexpr std::array<GraphCase, 7> kGraphCases = {{
    {"beamformer.yml", 3, 1},
    {"channelizer.yml", 3, 1},
    {"correlator.yml", 6, 2},
    {"detector.yml", 3, 1},
    {"integrator.yml", 6, 3},
    {"polarizer.yml", 3, 1},
    {"stacker.yml", 6, 3},
}};

struct TimedCycle {
    U64 block;
    bool emits;
};

struct TimedGraphCase {
    const char* file;
    U64 blocksPerWindow;
    std::vector<TimedCycle> cycles;
};

constexpr U64 kTimedBlockDuration = 16;

const std::vector<TimedGraphCase> kTimedGraphCases = {
    {"correlator.yml", 2, {{1, false}, {2, false}, {3, true}, {2, false}, {5, false},
                           {6, false}, {7, true}, {8, false}, {10, false}, {11, true}}},
    {"integrator.yml", 3, {{1, false}, {2, false}, {3, false}, {4, false}, {5, true},
                           {3, false}, {6, false}, {7, false}, {9, false}, {10, false},
                           {11, true}}},
    {"stacker.yml", 3, {{1, false}, {2, false}, {3, false}, {4, false}, {5, true},
                        {3, false}, {6, false}, {7, false}, {9, false}, {10, false},
                        {11, true}}},
};

struct FlowgraphGuard {
    Flowgraph flowgraph;
    bool created = false;

    ~FlowgraphGuard() {
        destroy();
    }

    Result destroy() {
        if (!created) {
            return Result::SUCCESS;
        }

        Result result = Result::SUCCESS;
        std::vector<std::string> names;
        if (flowgraph.view().keys(names) != Result::SUCCESS) {
            result = Result::ERROR;
        } else {
            for (const auto& name : names) {
                if (flowgraph.blockDestroy(name, false) != Result::SUCCESS) {
                    result = Result::ERROR;
                }
            }
        }
        if (flowgraph.destroy() != Result::SUCCESS) {
            result = Result::ERROR;
        }
        created = false;
        return result;
    }
};

#if defined(STELLINE_DEVICE_CPU_AVAILABLE) && defined(STELLINE_DEVICE_CUDA_AVAILABLE) && \
    defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
void ImportGraph(FlowgraphGuard& graph, const char* file) {
    REQUIRE(graph.flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    graph.created = true;
    const std::string path = std::string(STELLINE_TEST_FLOWGRAPH_DIR) + "/" + file;
    REQUIRE(graph.flowgraph.importFromFile(path) == Result::SUCCESS);
}

Tensor BlockOutput(FlowgraphGuard& graph, const std::string& block, const std::string& output) {
    Flowgraph::View::BlockData data;
    REQUIRE(graph.flowgraph.view().block(block, data) == Result::SUCCESS);
    return data.outputs.at(output).tensor;
}

void RequireTiming(const Tensor& tensor, const U64 timestamp, const U64 duration) {
    REQUIRE(std::any_cast<U64>(tensor.attribute("timestamp")) == timestamp);
    REQUIRE(std::any_cast<U64>(tensor.attribute("duration")) == duration);
}

std::string Metric(const Flowgraph::View::BlockData& block, const std::string& name) {
    const auto metric = std::find_if(block.metrics.begin(), block.metrics.end(),
                                     [&name](const auto& entry) {
                                         return entry.name == name;
                                     });
    REQUIRE(metric != block.metrics.end());
    return std::any_cast<std::string>(metric->value);
}

void MarkComparisonPending(Tensor& error) {
    REQUIRE(error.size() > 0);
    REQUIRE((error.dtype() == DataType::F32 || error.dtype() == DataType::F64));

    if (error.dtype() == DataType::F32) {
        error.data<F32>()[0] = std::numeric_limits<F32>::quiet_NaN();
    } else {
        error.data<F64>()[0] = std::numeric_limits<F64>::quiet_NaN();
    }
}

bool ComparisonCompleted(const Tensor& error) {
    if (error.dtype() == DataType::F32) {
        return std::isfinite(error.data<F32>()[0]);
    }
    if (error.dtype() == DataType::F64) {
        return std::isfinite(error.data<F64>()[0]);
    }
    return false;
}
#endif

}  // namespace

TEST_CASE("Reference flowgraphs on CPU match CUDA", "[flowgraph][cuda][parity]") {
#if !defined(STELLINE_DEVICE_CPU_AVAILABLE) || !defined(STELLINE_DEVICE_CUDA_AVAILABLE) || \
    !defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
    SKIP("Stelline was not built with both CPU and CUDA support.");
#else
    BackendGuard backendGuard;
    Backend::Config backendConfig;
    backendConfig.headless = true;
    REQUIRE(Backend::Initialize<DeviceType::CPU>(backendConfig) == Result::SUCCESS);
    REQUIRE(Backend::Initialize<DeviceType::CUDA>(backendConfig) == Result::SUCCESS);

    for (const auto& graphCase : kGraphCases) {
        INFO("Flowgraph: " << graphCase.file);

        FlowgraphGuard graph;
        ImportGraph(graph, graphCase.file);

        Flowgraph::View::BlockData comparator;
        REQUIRE(graph.flowgraph.view().block("compare", comparator) == Result::SUCCESS);
        Tensor comparisonError = comparator.outputs.at("error").tensor;

        for (U64 cycle = 0; cycle < graphCase.cycles; ++cycle) {
            INFO("Cycle: " << cycle);
            const bool expectsEmission = (cycle + 1) % graphCase.emissionPeriod == 0;
            if (expectsEmission) {
                MarkComparisonPending(comparisonError);
            }

            REQUIRE(graph.flowgraph.compute() == Result::SUCCESS);

            REQUIRE(graph.flowgraph.view().block("compare", comparator) == Result::SUCCESS);
            INFO(comparator.diagnostic);
            REQUIRE(comparator.state == Block::State::Created);
            if (expectsEmission) {
                REQUIRE(ComparisonCompleted(comparisonError));
                REQUIRE(graph.flowgraph.view().metrics("compare", comparator.metrics) ==
                        Result::SUCCESS);
                INFO("Max difference: " << Metric(comparator, "maxDiff"));
                REQUIRE(Metric(comparator, "match") == "PASS");
            }
        }

        REQUIRE(graph.destroy() == Result::SUCCESS);
    }
#endif
}

TEST_CASE("Timed reference flowgraphs on CPU match CUDA windows",
          "[flowgraph][cuda][parity][timing]") {
#if !defined(STELLINE_DEVICE_CPU_AVAILABLE) || !defined(STELLINE_DEVICE_CUDA_AVAILABLE) || \
    !defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
    SKIP("Stelline was not built with both CPU and CUDA support.");
#else
    BackendGuard backendGuard;
    Backend::Config backendConfig;
    backendConfig.headless = true;
    REQUIRE(Backend::Initialize<DeviceType::CPU>(backendConfig) == Result::SUCCESS);
    REQUIRE(Backend::Initialize<DeviceType::CUDA>(backendConfig) == Result::SUCCESS);

    for (const auto& graphCase : kTimedGraphCases) {
        INFO("Flowgraph: " << graphCase.file);

        FlowgraphGuard graph;
        ImportGraph(graph, graphCase.file);
        Tensor source = BlockOutput(graph, "source", "signal");
        Tensor comparisonError = BlockOutput(graph, "compare", "error");

        for (const auto& cycle : graphCase.cycles) {
            INFO("Block: " << cycle.block);
            REQUIRE(source.setAttribute("timestamp", cycle.block * kTimedBlockDuration) ==
                    Result::SUCCESS);
            REQUIRE(source.setAttribute("duration", kTimedBlockDuration) == Result::SUCCESS);
            MarkComparisonPending(comparisonError);

            REQUIRE(graph.flowgraph.compute() == Result::SUCCESS);

            Flowgraph::View::BlockData comparator;
            REQUIRE(graph.flowgraph.view().block("compare", comparator) == Result::SUCCESS);
            INFO(comparator.diagnostic);
            REQUIRE(comparator.state == Block::State::Created);
            REQUIRE(ComparisonCompleted(comparisonError) == cycle.emits);
            if (cycle.emits) {
                REQUIRE(graph.flowgraph.view().metrics("compare", comparator.metrics) ==
                        Result::SUCCESS);
                INFO("Max difference: " << Metric(comparator, "maxDiff"));
                REQUIRE(Metric(comparator, "match") == "PASS");

                const U64 firstBlock = cycle.block + 1 - graphCase.blocksPerWindow;
                const U64 duration = graphCase.blocksPerWindow * kTimedBlockDuration;
                RequireTiming(BlockOutput(graph, "cpu_dut", "buffer"),
                              firstBlock * kTimedBlockDuration, duration);
                RequireTiming(BlockOutput(graph, "cuda_dut", "buffer"),
                              firstBlock * kTimedBlockDuration, duration);
            }
        }

        REQUIRE(graph.destroy() == Result::SUCCESS);
    }
#endif
}
