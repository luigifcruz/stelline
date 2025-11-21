#include <stelline/types.hh>
#include <stelline/operators/filesystem/base.hh>
#include <fmt/format.h>

#include "utils/helpers.hh"

using namespace gxf;
using namespace holoscan;

namespace stelline::operators::filesystem {

struct DummyWriterOp::Impl {
    // State.

    std::chrono::time_point<std::chrono::system_clock> lastTime;
    std::chrono::milliseconds duration;
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
    pimpl->duration = std::chrono::milliseconds(0);
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

    // Measure time between messages.

    if (pimpl->lastTime.time_since_epoch().count() != 0) {
        auto now = std::chrono::system_clock::now();
        pimpl->duration += std::chrono::duration_cast<std::chrono::milliseconds>(now - pimpl->lastTime);
    }

    // Print statistics.

    if (pimpl->numIterations++ % 100 == 0) {
        pimpl->duration = std::chrono::milliseconds(0);
    }

    // Reset timer.

    pimpl->lastTime = std::chrono::system_clock::now();
}

stelline::StoreInterface::MetricsMap DummyWriterOp::collectMetricsMap() {
    if (!pimpl) {
        return {};
    }
    stelline::StoreInterface::MetricsMap metrics;
    metrics["iterations"] = fmt::format("{}", pimpl->numIterations);
    metrics["average_duration_ms"] = fmt::format("{}", pimpl->duration.count() / 100);
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
