#include <fcntl.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <stelline/types.hh>
#include <stelline/operators/filesystem/base.hh>
#include <fmt/format.h>

#include "utils/helpers.hh"
#include "utils/modifiers.hh"

using namespace gxf;
using namespace holoscan;

namespace stelline::operators::filesystem {

struct SimpleWriterOp::Impl {
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
};

void SimpleWriterOp::initialize() {
    // Allocate memory.
    pimpl = new Impl();

    // Initialize operator.
    Operator::initialize();
}

SimpleWriterOp::~SimpleWriterOp() {
    delete pimpl;
}

void SimpleWriterOp::setup(OperatorSpec& spec) {
    spec.input<std::shared_ptr<holoscan::Tensor>>("in")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));

    spec.param(filePath_, "file_path");
}

void SimpleWriterOp::start() {
    // Convert Parameters to variables.

    pimpl->filePath = filePath_.get();

    // Open Unix file.
    pimpl->file.open(pimpl->filePath.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);

    // Create stream.
    cudaStreamCreateWithFlags(&pimpl->stream, cudaStreamNonBlocking);

    // Reset counters.

    pimpl->bytesWritten = 0;
    pimpl->bytesSinceLastMeasurement = 0;
    pimpl->lastMeasurementTime = std::chrono::steady_clock::now();
    pimpl->currentBandwidthMBps = 0.0;

    HOLOSCAN_LOG_INFO("Successfully opened file '{}'.", pimpl->filePath);
}

void SimpleWriterOp::stop() {
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

void SimpleWriterOp::compute(InputContext& input, OutputContext&, ExecutionContext&) {
    const auto& tensor = input.receive<std::shared_ptr<holoscan::Tensor>>("in").value();
    const auto& tensorBytes = tensor->size() * (tensor->dtype().bits / 8);

    // Allocate permuted tensor.

    if (pimpl->bytesWritten == 0) {
        CUDA_CHECK_THROW(BlockAlloc(tensor, pimpl->permutedTensor), [&]{
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

    CUDA_CHECK_THROW(BlockPermutation(pimpl->permutedTensor->to_dlpack(), tensor->to_dlpack()), [&]{
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

stelline::StoreInterface::MetricsMap SimpleWriterOp::collectMetricsMap() {
    if (!pimpl) {
        return {};
    }
    auto now = std::chrono::steady_clock::now();
    auto elapsedSeconds = std::chrono::duration<double>(now - pimpl->lastMeasurementTime).count();

    if (elapsedSeconds > 0.0) {
        int64_t bytes = pimpl->bytesSinceLastMeasurement.exchange(0);
        pimpl->currentBandwidthMBps = static_cast<double>(bytes) / (1024.0 * 1024.0) / elapsedSeconds;
        pimpl->lastMeasurementTime = now;
    }

    stelline::StoreInterface::MetricsMap metrics;
    metrics["current_bandwidth_mb_s"] = fmt::format("{:.2f}", pimpl->currentBandwidthMBps.load());
    metrics["total_data_written_mb"] = fmt::format("{:.0f}", static_cast<double>(pimpl->bytesWritten) / (1024.0 * 1024.0));
    return metrics;
}

std::string SimpleWriterOp::collectMetricsString() {
    if (!pimpl) {
        return {};
    }
    const auto metrics = collectMetricsMap();
    return fmt::format("  Current Bandwidth: {} MB/s\n"
                       "  Total Data Written: {} MB",
                       metrics.at("current_bandwidth_mb_s"),
                       metrics.at("total_data_written_mb"));
}

}  // namespace stelline::operators::filesystem
