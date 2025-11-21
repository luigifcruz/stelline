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
    spec.input<std::shared_ptr<holoscan::Tensor>>("in")
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
    // Deallocate host bounce buffer.

    if (pimpl->bounceBuffer != nullptr) {
        cudaFreeHost(pimpl->bounceBuffer);
    }

    // Destroy CUDA stream.
    cudaStreamDestroy(pimpl->stream);

    HOLOSCAN_LOG_INFO("Successfully closed ZeroMQ transmitter.");
}

void ZmqTransmitterOp::compute(InputContext& input, OutputContext&, ExecutionContext&) {
    const auto& tensor = input.receive<std::shared_ptr<holoscan::Tensor>>("in").value();
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

stelline::StoreInterface::MetricsMap ZmqTransmitterOp::collectMetricsMap() {
    auto now = std::chrono::steady_clock::now();
    auto elapsedSeconds = std::chrono::duration<double>(now - pimpl->lastMeasurementTime).count();

    if (elapsedSeconds > 0.0) {
        int64_t bytes = pimpl->bytesSinceLastMeasurement.exchange(0);
        pimpl->currentBandwidthMBps = static_cast<double>(bytes) / (1024.0 * 1024.0) / elapsedSeconds;
        pimpl->lastMeasurementTime = now;
    }

    stelline::StoreInterface::MetricsMap metrics;
    metrics["current_bandwidth_mb_s"] = fmt::format("{:.2f}", pimpl->currentBandwidthMBps.load());
    metrics["total_bytes_written"] = fmt::format("{}", pimpl->bytesWritten);
    return metrics;
}

std::string ZmqTransmitterOp::collectMetricsString() {
    const auto metrics = collectMetricsMap();
    return fmt::format("ZeroMQ Transmitter Operator:\n"
                       "  Input Bandwidth: {} MB/s\n"
                       "  Total Bytes Written: {}",
                       metrics.at("current_bandwidth_mb_s"),
                       metrics.at("total_bytes_written"));
}

}  // namespace stelline::operators::socket
