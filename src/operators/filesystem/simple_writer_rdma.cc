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

struct SimpleWriterRdmaOp::Impl {
    // State.

    std::string filePath;
    int fileDescriptor;
    int64_t bytesWritten;
    CUfileHandle_t cufileHandle;
    std::shared_ptr<holoscan::Tensor> permutedTensor;

    // Metrics.

    std::chrono::time_point<std::chrono::steady_clock> lastMeasurementTime;
    std::atomic<int64_t> bytesSinceLastMeasurement{0};
    std::atomic<double> currentBandwidthMBps{0.0};
};

void SimpleWriterRdmaOp::initialize() {
    // Allocate memory.
    pimpl = new Impl();

    // Initialize operator.
    Operator::initialize();
}

SimpleWriterRdmaOp::~SimpleWriterRdmaOp() {
    delete pimpl;
}

void SimpleWriterRdmaOp::setup(OperatorSpec& spec) {
    spec.input<std::shared_ptr<holoscan::Tensor>>("in")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));

    spec.param(filePath_, "file_path");
}

void SimpleWriterRdmaOp::start() {
    // Convert Parameters to variables.

    pimpl->filePath = filePath_.get();

    // Open Unix file.

    pimpl->fileDescriptor = open(pimpl->filePath.c_str(), O_CREAT | O_WRONLY | O_DIRECT | O_TRUNC, 0644);
    if (pimpl->fileDescriptor < 0) {
        HOLOSCAN_LOG_ERROR("Unable to open file '{}'. Error number '{}'.",
            pimpl->filePath, pimpl->fileDescriptor);
    }

    // Initialize GDS driver.

    GDS_CHECK_THROW(cuFileDriverOpen(), [&]{
        HOLOSCAN_LOG_ERROR("Failed to initialize GDS driver.");
        close(pimpl->fileDescriptor);
    });

    // Registering file descriptor with GDS driver.

    CUfileDescr_t cufileDescriptor = {};
    cufileDescriptor.handle.fd = pimpl->fileDescriptor;
    cufileDescriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    GDS_CHECK_THROW(cuFileHandleRegister(&pimpl->cufileHandle, &cufileDescriptor), [&]{
        HOLOSCAN_LOG_ERROR("Failed to register file descriptor with GDS driver.");
        close(pimpl->fileDescriptor);
    });

    // Reset counters.

    pimpl->bytesWritten = 0;
    pimpl->bytesSinceLastMeasurement = 0;
    pimpl->lastMeasurementTime = std::chrono::steady_clock::now();
    pimpl->currentBandwidthMBps = 0.0;

    HOLOSCAN_LOG_INFO("Successfully opened file '{}'.", pimpl->filePath);
}

void SimpleWriterRdmaOp::stop() {
    // Deregister buffer with GDS driver.

    if (pimpl->bytesWritten != 0) {
        GDS_CHECK_THROW(cuFileBufDeregister(pimpl->permutedTensor->data()), [&]{
            HOLOSCAN_LOG_ERROR("Failed to deregister buffer with GDS driver.");
        });
    }

    // Deregister file descriptor.
    cuFileHandleDeregister(&pimpl->cufileHandle);

    // Close file descriptor.
    close(pimpl->fileDescriptor);

    // Close GDS driver.
    cuFileDriverClose();

    HOLOSCAN_LOG_INFO("Successfully closed file '{}'.", pimpl->filePath);
}

void SimpleWriterRdmaOp::compute(InputContext& input, OutputContext&, ExecutionContext&) {
    const auto& tensor = input.receive<std::shared_ptr<holoscan::Tensor>>("in").value();
    const auto& tensorBytes = tensor->size() * (tensor->dtype().bits / 8);

    // Allocate permuted tensor.

    if (pimpl->bytesWritten == 0) {
        CUDA_CHECK_THROW(BlockAlloc(tensor, pimpl->permutedTensor), [&]{
            HOLOSCAN_LOG_ERROR("Failed to allocate permuted tensor.");
        });

        GDS_CHECK_THROW(cuFileBufRegister(pimpl->permutedTensor->data(), tensorBytes, 0), [&]{
            HOLOSCAN_LOG_ERROR("Failed to register buffer with GDS driver.");
        });
    }

    // Permute tensor.

    CUDA_CHECK_THROW(BlockPermutation(pimpl->permutedTensor->to_dlpack(), tensor->to_dlpack()), [&]{
        HOLOSCAN_LOG_ERROR("Failed to permute tensor.");
    });

    // Write tensor directly to file.

    const auto res = cuFileWrite(pimpl->cufileHandle,
                                 pimpl->permutedTensor->data(),
                                 tensorBytes,
                                 pimpl->bytesWritten,
                                 0);
    if (res != tensorBytes) {
        HOLOSCAN_LOG_ERROR("Failed to write to file. Expected {} bytes, but wrote {}. Errno {}.", tensorBytes, res, errno);
        throw std::runtime_error("Failed to write to file.");
    }

    pimpl->bytesWritten += tensorBytes;
    pimpl->bytesSinceLastMeasurement += tensorBytes;
}

stelline::StoreInterface::MetricsMap SimpleWriterRdmaOp::collectMetricsMap() {
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

std::string SimpleWriterRdmaOp::collectMetricsString() {
    const auto metrics = collectMetricsMap();
    return fmt::format("Simple Writer RDMA Operator:\n"
                       "  Current Bandwidth: {} MB/s\n"
                       "  Total Data Written: {} MB",
                       metrics.at("current_bandwidth_mb_s"),
                       metrics.at("total_data_written_mb"));
}

}  // namespace stelline::operators::filesystem
