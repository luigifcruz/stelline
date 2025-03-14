#include <stelline/types.hh>
#include <stelline/operators/io/base.hh>

#include "helpers.hh"

using namespace gxf;
using namespace holoscan;

namespace stelline::operators::io {

struct DummySinkOp::Impl {
    // State.

    std::chrono::time_point<std::chrono::system_clock> lastTime;
    std::chrono::milliseconds duration;
    uint64_t numIterations;

    // Metrics.

    std::thread metricsThread;
    bool metricsThreadRunning;
    void metricsLoop();
};

void DummySinkOp::initialize() {
    // Allocate memory.
    pimpl = new Impl();

    // Initialize operator.
    Operator::initialize();
}

DummySinkOp::~DummySinkOp() {
    delete pimpl;
}

void DummySinkOp::setup(OperatorSpec& spec) {
    spec.input<DspBlock>("in")
        .connector(IOSpec::ConnectorType::kDoubleBuffer, 
                   holoscan::Arg("capacity", 1024UL));
}

void DummySinkOp::start() {
    // Start metrics thread.

    pimpl->metricsThreadRunning = true;
    pimpl->metricsThread = std::thread([&]{
        pimpl->metricsLoop();
    });
}

void DummySinkOp::stop() {
    // Stop metrics thread.

    pimpl->metricsThreadRunning = false;
    if (pimpl->metricsThread.joinable()) {
        pimpl->metricsThread.join();
    }
}

void DummySinkOp::compute(InputContext& input, OutputContext&, ExecutionContext&) {
    // Receive tensor.

    input.receive<std::shared_ptr<int>>("in");

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

void DummySinkOp::Impl::metricsLoop() {
    while (metricsThreadRunning) {
        HOLOSCAN_LOG_INFO("Dummy Sink Operator:");
        HOLOSCAN_LOG_INFO("  Iterations      : {}", numIterations);
        HOLOSCAN_LOG_INFO("  Average Duration: {} ms", duration.count() / 100);

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

}  // namespace stelline::operators::io
