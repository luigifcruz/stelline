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

// TODO: Implement asynchronous GDS writes.

struct SimpleSinkOp::Impl {
    // File Implementation

    struct {
        int fileDescriptor;
        int64_t bytesWritten;
        CUfileHandle_t cufileHandle;
        uint64_t bufferedRegistrations = 0;
        std::unordered_set<void*> registeredBuffers;
    } Gds;

    struct {
        void* bounceBuffer;
        cudaStream_t stream;
        std::ofstream file;
    } Unix;

    // State.

    std::string filePath;
    bool enableRdma;

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
    spec.param(enableRdma_, "enable_rdma");
}

void SimpleSinkOp::start() {
    // Convert Parameters to variables.

    pimpl->filePath = filePath_.get();
    pimpl->enableRdma = enableRdma_.get();

    // Choose File Implementation

    if (pimpl->enableRdma) {
        // Open Unix file.

        pimpl->Gds.fileDescriptor = open(pimpl->filePath.c_str(), O_CREAT | O_WRONLY | O_DIRECT, 0644);
        if (pimpl->Gds.fileDescriptor < 0) {
            HOLOSCAN_LOG_ERROR("Unable to open file '{}'. Error number '{}'.", 
                pimpl->filePath, pimpl->Gds.fileDescriptor);
        }

        // Initialize GDS driver.

        GDS_CHECK_THROW(cuFileDriverOpen(), [&]{
            HOLOSCAN_LOG_ERROR("Failed to initialize GDS driver.");
            close(pimpl->Gds.fileDescriptor);
        });

        // Registering file descriptor with GDS driver.

        CUfileDescr_t cufileDescriptor = {};
        cufileDescriptor.handle.fd = pimpl->Gds.fileDescriptor;
        cufileDescriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        GDS_CHECK_THROW(cuFileHandleRegister(&pimpl->Gds.cufileHandle, &cufileDescriptor), [&]{
            HOLOSCAN_LOG_ERROR("Failed to register file descriptor with GDS driver.");
            close(pimpl->Gds.fileDescriptor);
        });

        // Reset counters.

        pimpl->Gds.bytesWritten = 0;
    } else {
        // Open Unix file.
        pimpl->Unix.file.open(pimpl->filePath.c_str(), std::ios::out | std::ios::binary);

        // Create stream.
        cudaStreamCreateWithFlags(&pimpl->Unix.stream, cudaStreamNonBlocking);
    }

    HOLOSCAN_LOG_INFO("Successfully opened file '{}'.", pimpl->filePath);

    // Start metrics thread.

    pimpl->metricsThreadRunning = true;
    pimpl->metricsThread = std::thread([&]{
        pimpl->metricsLoop();
    });
}

void SimpleSinkOp::stop() {
    // Stop metrics thread.

    pimpl->metricsThreadRunning = false;
    if (pimpl->metricsThread.joinable()) {
        pimpl->metricsThread.join();
    } 

    // Close file.

    if (pimpl->enableRdma) {
        // Deregister buffers with GDS driver.

        for (const auto& buffer : pimpl->Gds.registeredBuffers) {
            GDS_CHECK_THROW(cuFileBufDeregister(buffer), [&]{
                HOLOSCAN_LOG_ERROR("Failed to deregister buffer with GDS driver.");
            });
        }

        // Deregister file descriptor.
        cuFileHandleDeregister(&pimpl->Gds.cufileHandle);

        // Close file descriptor.
        close(pimpl->Gds.fileDescriptor);

        // Close GDS driver.
        cuFileDriverClose();
    } else {
        // Deallocate host bounce buffer.

        if (pimpl->Unix.bounceBuffer != nullptr) {
            cudaFreeHost(pimpl->Unix.bounceBuffer);
        }

        // Destroy CUDA stream.
        cudaStreamDestroy(pimpl->Unix.stream);

        // Close Unix file.
        pimpl->Unix.file.close();
    }

    HOLOSCAN_LOG_INFO("Successfully closed file '{}'.", pimpl->filePath);
}

void SimpleSinkOp::compute(InputContext& input, OutputContext&, ExecutionContext&) {
    const auto& tensor = input.receive<DspBlock>("in").value().tensor;
    const auto& tensorBytes = tensor->size() * (tensor->dtype().bits / 8);

    // Write tensor to file.

    if (pimpl->enableRdma) {
        // Register buffer with GDS driver.

        if (!pimpl->Gds.registeredBuffers.contains(tensor->data())) {
            pimpl->Gds.registeredBuffers.insert(tensor->data());

            GDS_CHECK_THROW(cuFileBufRegister(tensor->data(), tensorBytes, 0), [&]{
                HOLOSCAN_LOG_ERROR("Failed to register buffer with GDS driver.");
            });
        } else {
            pimpl->Gds.bufferedRegistrations++;
        }

        // Write tensor directly to file.

        const auto res = cuFileWrite(pimpl->Gds.cufileHandle, 
                                     tensor->data(), 
                                     tensorBytes,
                                     pimpl->Gds.bytesWritten, 
                                     0);
        if (res != tensorBytes) {
            HOLOSCAN_LOG_ERROR("Failed to write to file. Expected {} bytes, but wrote {}. Errno {}.", tensorBytes, res, errno); 
            throw std::runtime_error("Failed to write to file.");
        }
        pimpl->Gds.bytesWritten += tensorBytes;
    } else {
        // Allocate host bounce buffer.

        if (pimpl->Unix.bounceBuffer == nullptr) {
            if (cudaMallocHost(&pimpl->Unix.bounceBuffer, tensorBytes) != cudaSuccess) {
                HOLOSCAN_LOG_ERROR("Failed to allocate pinned host bounce buffer.");
                throw std::runtime_error("Failed to write to file.");
            }
        }

        // Transfer tensor to host.

        cudaMemcpyAsync(pimpl->Unix.bounceBuffer, 
                        tensor->data(), 
                        tensorBytes, 
                        cudaMemcpyDeviceToHost, 
                        pimpl->Unix.stream);
        cudaStreamSynchronize(pimpl->Unix.stream);

        // Write to file.
        
        pimpl->Unix.file.write(reinterpret_cast<char*>(pimpl->Unix.bounceBuffer), tensorBytes);
        pimpl->Unix.file.flush();
    }
}

void SimpleSinkOp::Impl::metricsLoop() {
    while (metricsThreadRunning) {
        HOLOSCAN_LOG_INFO("Simple Sink Operator:");
        if (enableRdma) {
            HOLOSCAN_LOG_INFO("  GDS Registered Buffers: {}", Gds.registeredBuffers.size());
            HOLOSCAN_LOG_INFO("  GDS Buffered Registrations: {}", Gds.bufferedRegistrations);
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

}  // namespace stelline::operators::io
