#include <stelline/types.hh>
#include <stelline/operators/filesystem/base.hh>
#include <fmt/format.h>

#include "utils/helpers.hh"

using namespace gxf;
using namespace holoscan;

namespace stelline::operators::filesystem {

struct DummyWriterOp::Impl {
    // State.

    std::chrono::time_point<std::chrono::steady_clock> lastTime;
    std::chrono::time_point<std::chrono::steady_clock> startTime;
    uint64_t numIterations;

    // Metrics.

    uint64_t latestTimestamp;
};

void DummyWriterOp::initialize() {
    // Allocate memory.
    pimpl = new Impl();

    // Initialize operator.
    Operator::initialize();
}

DummyWriterOp::~DummyWriterOp() {
    delete pimpl;
}

void DummyWriterOp::setup(OperatorSpec& spec) {
    spec.input<std::shared_ptr<holoscan::Tensor>>("in")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));
}

void DummyWriterOp::start() {
    pimpl->numIterations = 0;
    pimpl->startTime = std::chrono::steady_clock::now();
    pimpl->lastTime = {};
    pimpl->latestTimestamp = 0;
}

void DummyWriterOp::stop() {
}

void DummyWriterOp::compute(InputContext& input, OutputContext&, ExecutionContext&) {
    // Receive tensor.

    input.receive<std::shared_ptr<holoscan::Tensor>>("in");

    // Log latest timestamp.

    const auto& meta = metadata();
    pimpl->latestTimestamp = meta->get<uint64_t>("timestamp");

    // Increment iteration counter.

    pimpl->numIterations++;
}

stelline::MetricsInterface::MetricsMap DummyWriterOp::collectMetricsMap() {
    if (!pimpl) {
        return {};
    }

    auto elapsed = std::chrono::steady_clock::now() - pimpl->startTime;
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    double avgMs = (pimpl->numIterations > 0) ? static_cast<double>(elapsedMs) / pimpl->numIterations : 0.0;

    stelline::MetricsInterface::MetricsMap metrics;
    metrics["iterations"] = fmt::format("{}", pimpl->numIterations);
    metrics["average_duration_ms"] = fmt::format("{:.2f}", avgMs);
    metrics["latest_timestamp"] = fmt::format("{}", pimpl->latestTimestamp);
    return metrics;
}

std::string DummyWriterOp::collectMetricsString() {
    if (!pimpl) {
        return {};
    }
    const auto metrics = collectMetricsMap();
    return fmt::format("  Iterations      : {}\n"
                       "  Average Duration: {} ms\n"
                       "  Latest Timestamp: {}",
                       metrics.at("iterations"),
                       metrics.at("average_duration_ms"),
                       metrics.at("latest_timestamp"));
}

}  // namespace stelline::operators::io
