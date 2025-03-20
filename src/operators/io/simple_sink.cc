#include <fcntl.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <stelline/types.hh>
#include <stelline/operators/io/base.hh>

#include "helpers.hh"

using namespace gxf;
using namespace holoscan;

namespace stelline::operators::io {

struct SimpleSinkOp::Impl {
    // State.

    std::string filePath;
    void* bounceBuffer;
    cudaStream_t stream;
    std::ofstream file;

    // Metrics.

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

    // Allocate host bounce buffer.

    if (pimpl->bounceBuffer == nullptr) {
        if (cudaMallocHost(&pimpl->bounceBuffer, tensorBytes) != cudaSuccess) {
            HOLOSCAN_LOG_ERROR("Failed to allocate pinned host bounce buffer.");
            throw std::runtime_error("Failed to write to file.");
        }
    }

    // Transfer tensor to host.

    cudaMemcpyAsync(pimpl->bounceBuffer,
                    tensor->data(),
                    tensorBytes,
                    cudaMemcpyDeviceToHost,
                    pimpl->stream);
    cudaStreamSynchronize(pimpl->stream);

    // Write to file.

    pimpl->file.write(reinterpret_cast<char*>(pimpl->bounceBuffer), tensorBytes);
    pimpl->file.flush();
}

void SimpleSinkOp::Impl::metricsLoop() {
    while (metricsThreadRunning) {
        HOLOSCAN_LOG_INFO("Simple Sink Operator:");
        HOLOSCAN_LOG_INFO("  N/A");

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

}  // namespace stelline::operators::io
