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

struct SimpleSinkRdmaOp::Impl {
    // State.

    std::string filePath;
    int fileDescriptor;
    int64_t bytesWritten;
    CUfileHandle_t cufileHandle;
    uint64_t bufferedRegistrations = 0;
    std::unordered_set<void*> registeredBuffers;

    // Metrics.

    std::thread metricsThread;
    bool metricsThreadRunning;
    void metricsLoop();
};

void SimpleSinkRdmaOp::initialize() {
    // Allocate memory.
    pimpl = new Impl();

    // Initialize operator.
    Operator::initialize();
}

SimpleSinkRdmaOp::~SimpleSinkRdmaOp() {
    delete pimpl;
}

void SimpleSinkRdmaOp::setup(OperatorSpec& spec) {
    spec.input<DspBlock>("in")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));

    spec.param(filePath_, "file_path");
}

void SimpleSinkRdmaOp::start() {
    // Convert Parameters to variables.

    pimpl->filePath = filePath_.get();

    // Open Unix file.

    pimpl->fileDescriptor = open(pimpl->filePath.c_str(), O_CREAT | O_WRONLY | O_DIRECT, 0644);
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

    // Start metrics thread.

    pimpl->metricsThreadRunning = true;
    pimpl->metricsThread = std::thread([&]{
        pimpl->metricsLoop();
    });

    HOLOSCAN_LOG_INFO("Successfully opened file '{}'.", pimpl->filePath);
}

void SimpleSinkRdmaOp::stop() {
    // Stop metrics thread.

    pimpl->metricsThreadRunning = false;
    if (pimpl->metricsThread.joinable()) {
        pimpl->metricsThread.join();
    }

    // Deregister buffers with GDS driver.

    for (const auto& buffer : pimpl->registeredBuffers) {
        GDS_CHECK_THROW(cuFileBufDeregister(buffer), [&]{
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

void SimpleSinkRdmaOp::compute(InputContext& input, OutputContext&, ExecutionContext&) {
    const auto& tensor = input.receive<DspBlock>("in").value().tensor;
    const auto& tensorBytes = tensor->size() * (tensor->dtype().bits / 8);

    // Register buffer with GDS driver.

    if (!pimpl->registeredBuffers.contains(tensor->data())) {
        pimpl->registeredBuffers.insert(tensor->data());

        GDS_CHECK_THROW(cuFileBufRegister(tensor->data(), tensorBytes, 0), [&]{
            HOLOSCAN_LOG_ERROR("Failed to register buffer with GDS driver.");
        });
    } else {
        pimpl->bufferedRegistrations++;
    }

    // Write tensor directly to file.

    const auto res = cuFileWrite(pimpl->cufileHandle,
                                 tensor->data(),
                                 tensorBytes,
                                 pimpl->bytesWritten,
                                 0);
    if (res != tensorBytes) {
        HOLOSCAN_LOG_ERROR("Failed to write to file. Expected {} bytes, but wrote {}. Errno {}.", tensorBytes, res, errno);
        throw std::runtime_error("Failed to write to file.");
    }
    pimpl->bytesWritten += tensorBytes;
}

void SimpleSinkRdmaOp::Impl::metricsLoop() {
    while (metricsThreadRunning) {
        HOLOSCAN_LOG_INFO("Simple Sink RDMA Operator:");
        HOLOSCAN_LOG_INFO("  Registered Buffers: {}", registeredBuffers.size());
        HOLOSCAN_LOG_INFO("  Buffered Registrations: {}", bufferedRegistrations);

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

}  // namespace stelline::operators::io
