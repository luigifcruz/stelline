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

void DummyWriterOp::tick() {
    if (!pimpl || !metrics()) {
        return;
    }

    auto elapsed = std::chrono::steady_clock::now() - pimpl->startTime;
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    double avgMs = (pimpl->numIterations > 0) ? static_cast<double>(elapsedMs) / pimpl->numIterations : 0.0;

    metrics()->record("iterations", fmt::format("{}", pimpl->numIterations));
    metrics()->record("average_duration_ms", fmt::format("{:.2f}", avgMs));
    metrics()->record("latest_timestamp", fmt::format("{}", pimpl->latestTimestamp));
}

std::string DummyWriterOp::formatMetrics(const MetricsProvider::MetricsMap& metrics) {
    return fmt::format("  Iterations      : {}\n"
                       "  Average Duration: {} ms\n"
                       "  Latest Timestamp: {}",
                       metrics.at("iterations").value,
                       metrics.at("average_duration_ms").value,
                       metrics.at("latest_timestamp").value);
}

}  // namespace stelline::operators::io
