#include <fcntl.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <stelline/types.hh>
#include <stelline/operators/io/base.hh>

#include "helpers.hh"
#include "permute.hh"

using namespace gxf;
using namespace holoscan;

namespace stelline::operators::io {

struct SimpleSinkOp::Impl {
    // State.

    std::string filePath;
    int64_t bytesWritten;
    void* bounceBuffer;
    cudaStream_t stream;
    std::ofstream file;
    std::shared_ptr<holoscan::Tensor> permutedTensor;

    // Metrics.

    std::chrono::time_point<std::chrono::steady_clock> lastMeasurementTime;
    std::atomic<int64_t> bytesSinceLastMeasurement{0};
    std::atomic<double> currentBandwidthMBps{0.0};

    std::thread metricsThread;
    bool metricsThreadRunning;
    void metricsLoop();
};

void SimpleSinkOp::initialize() {
    // Allocate memory.
    pimpl = new Impl();

    // Initialize operator.
    Operator::initialize();
}

SimpleSinkOp::~SimpleSinkOp() {
    delete pimpl;
}

void SimpleSinkOp::setup(OperatorSpec& spec) {
    spec.input<DspBlock>("in")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));

    spec.param(filePath_, "file_path");
}

void SimpleSinkOp::start() {
    // Convert Parameters to variables.

    pimpl->filePath = filePath_.get();

    // Open Unix file.
    pimpl->file.open(pimpl->filePath.c_str(), std::ios::out | std::ios::binary);

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

    HOLOSCAN_LOG_INFO("Successfully opened file '{}'.", pimpl->filePath);
}

void SimpleSinkOp::stop() {
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

    // Close Unix file.
    pimpl->file.close();

    HOLOSCAN_LOG_INFO("Successfully closed file '{}'.", pimpl->filePath);
}

void SimpleSinkOp::compute(InputContext& input, OutputContext&, ExecutionContext&) {
    const auto& tensor = input.receive<DspBlock>("in").value().tensor;
    const auto& tensorBytes = tensor->size() * (tensor->dtype().bits / 8);

    // Allocate permuted tensor.

    if (pimpl->bytesWritten == 0) {
        CUDA_CHECK_THROW(DspBlockAlloc(tensor, pimpl->permutedTensor), [&]{
            HOLOSCAN_LOG_ERROR("Failed to allocate permuted tensor.");
        });
    }

    // Allocate host bounce buffer.

    if (pimpl->bounceBuffer == nullptr) {
        if (cudaMallocHost(&pimpl->bounceBuffer, tensorBytes) != cudaSuccess) {
            HOLOSCAN_LOG_ERROR("Failed to allocate pinned host bounce buffer.");
            throw std::runtime_error("Failed to write to file.");
        }
    }

    // Permute tensor.

    CUDA_CHECK_THROW(DspBlockPermutation(pimpl->permutedTensor->to_dlpack(), tensor->to_dlpack()), [&]{
        HOLOSCAN_LOG_ERROR("Failed to permute tensor.");
    });

    // Transfer tensor to host.

    cudaMemcpyAsync(pimpl->bounceBuffer,
                    pimpl->permutedTensor->data(),
                    tensorBytes,
                    cudaMemcpyDeviceToHost,
                    pimpl->stream);
    cudaStreamSynchronize(pimpl->stream);

    // Write to file.

    pimpl->file.write(reinterpret_cast<char*>(pimpl->bounceBuffer), tensorBytes);
    pimpl->file.flush();

    pimpl->bytesWritten += tensorBytes;
    pimpl->bytesSinceLastMeasurement += tensorBytes;
}

void SimpleSinkOp::Impl::metricsLoop() {
    while (metricsThreadRunning) {
        auto now = std::chrono::steady_clock::now();
        auto elapsedSeconds = std::chrono::duration<double>(now - lastMeasurementTime).count();

        if (elapsedSeconds > 0.0) {
            int64_t bytes = bytesSinceLastMeasurement.exchange(0);
            currentBandwidthMBps = static_cast<double>(bytes) / (1024.0 * 1024.0) / elapsedSeconds;
            lastMeasurementTime = now;
        }

        HOLOSCAN_LOG_INFO("Simple Sink Operator:");
        HOLOSCAN_LOG_INFO("  Current Bandwidth: {:.2f} MB/s", currentBandwidthMBps);
        HOLOSCAN_LOG_INFO("  Total Data Written: {:.0f} MB", static_cast<double>(bytesWritten) / (1024.0 * 1024.0));

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

}  // namespace stelline::operators::io
