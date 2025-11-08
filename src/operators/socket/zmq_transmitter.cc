#include <stelline/types.hh>
#include <stelline/operators/socket/base.hh>

#include <zmq.hpp>

using namespace gxf;
using namespace holoscan;

namespace stelline::operators::socket {

struct ZmqTransmitterOp::Impl {
    // State.

    std::string address;
    int64_t bytesWritten;
    void* bounceBuffer;
    cudaStream_t stream;

    zmq::context_t context;
    zmq::socket_t publisher;

    // Metrics.

    std::chrono::time_point<std::chrono::steady_clock> lastMeasurementTime;
    std::atomic<int64_t> bytesSinceLastMeasurement{0};
    std::atomic<double> currentBandwidthMBps{0.0};

    std::thread metricsThread;
    bool metricsThreadRunning;
    void metricsLoop();
};

void ZmqTransmitterOp::initialize() {
    // Allocate memory.
    pimpl = new Impl();

    // Initialize operator.
    Operator::initialize();
}

ZmqTransmitterOp::~ZmqTransmitterOp() {
    delete pimpl;
}

void ZmqTransmitterOp::setup(OperatorSpec& spec) {
    spec.input<DspBlock>("in")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));

    spec.param(address_, "address");
}

void ZmqTransmitterOp::start() {
    // Convert Parameters to variables.

    pimpl->address = address_.get();

    // Create stream.
    cudaStreamCreateWithFlags(&pimpl->stream, cudaStreamNonBlocking);

    // Reset counters.

    pimpl->bytesWritten = 0;
    pimpl->bytesSinceLastMeasurement = 0;
    pimpl->lastMeasurementTime = std::chrono::steady_clock::now();
    pimpl->currentBandwidthMBps = 0.0;

    // Start metrics thread.

    pimpl->metricsThreadRunning = true;
    pimpl->metricsThread = std::thread([&]{
        pimpl->metricsLoop();
    });

    // Initialize ZeroMQ context and socket.

    pimpl->context = zmq::context_t(1);
    pimpl->publisher = zmq::socket_t(pimpl->context, ZMQ_PUB);

    int hwm = 10000;
    int sndbuf = 1048576;
    pimpl->publisher.setsockopt(ZMQ_SNDHWM, &hwm, sizeof(hwm));
    pimpl->publisher.setsockopt(ZMQ_SNDBUF, &sndbuf, sizeof(sndbuf));
    pimpl->publisher.setsockopt(ZMQ_LINGER, 0);
    pimpl->publisher.bind(pimpl->address);

    HOLOSCAN_LOG_INFO("Successfully created ZeroMQ transmitter at address '{}'.", pimpl->address);
}

void ZmqTransmitterOp::stop() {
    // Stop metrics thread.

    pimpl->metricsThreadRunning = false;
    if (pimpl->metricsThread.joinable()) {
        pimpl->metricsThread.join();
    }

    // Deallocate host bounce buffer.

    if (pimpl->bounceBuffer != nullptr) {
        cudaFreeHost(pimpl->bounceBuffer);
    }

    // Destroy CUDA stream.
    cudaStreamDestroy(pimpl->stream);

    HOLOSCAN_LOG_INFO("Successfully closed ZeroMQ transmitter.");
}

void ZmqTransmitterOp::compute(InputContext& input, OutputContext&, ExecutionContext&) {
    const auto& tensor = input.receive<DspBlock>("in").value().tensor;
    const auto& tensorBytes = tensor->size() * (tensor->dtype().bits / 8);

    // Allocate host bounce buffer.

    if (pimpl->bounceBuffer == nullptr) {
        if (cudaMallocHost(&pimpl->bounceBuffer, tensorBytes) != cudaSuccess) {
            HOLOSCAN_LOG_ERROR("Failed to allocate pinned host bounce buffer.");
            throw std::runtime_error("Failed to allocate host bounce buffer.");
        }
    }

    // Transfer tensor to host.

    cudaMemcpyAsync(pimpl->bounceBuffer,
                    tensor->data(),
                    tensorBytes,
                    cudaMemcpyDeviceToHost,
                    pimpl->stream);
    cudaStreamSynchronize(pimpl->stream);

    // Send tensor to publisher.

    pimpl->publisher.send(pimpl->bounceBuffer, tensorBytes);

    pimpl->bytesWritten += tensorBytes;
    pimpl->bytesSinceLastMeasurement += tensorBytes;
}

void ZmqTransmitterOp::Impl::metricsLoop() {
    while (metricsThreadRunning) {
        auto now = std::chrono::steady_clock::now();
        auto elapsedSeconds = std::chrono::duration<double>(now - lastMeasurementTime).count();

        if (elapsedSeconds > 0.0) {
            int64_t bytes = bytesSinceLastMeasurement.exchange(0);
            currentBandwidthMBps = static_cast<double>(bytes) / (1024.0 * 1024.0) / elapsedSeconds;
            lastMeasurementTime = now;
        }

        HOLOSCAN_LOG_INFO("ZeroMQ Transmitter Operator:");
        HOLOSCAN_LOG_INFO("  Input Bandwidth: {:.2f} MB/s", currentBandwidthMBps.load());

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

}  // namespace stelline::operators::socket
