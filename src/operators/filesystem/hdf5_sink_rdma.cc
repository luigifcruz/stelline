#include <fcntl.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <stelline/types.hh>
#include <stelline/operators/filesystem/base.hh>

#include "helpers.hh"
#include "permute.hh"

using namespace gxf;
using namespace holoscan;

namespace stelline::operators::filesystem {

struct Hdf5SinkRdmaOp::Impl {
    // State.

    hid_t faplId, fileId, datasetId, dataspaceId, memspaceId, plistId;
    hid_t attributeId, attrDataspaceId, dimLabelsType;
    std::vector<hsize_t> dims;
    std::vector<hsize_t> maxdims;
    std::vector<hsize_t> chunkDims;
    std::vector<hsize_t> count;
    std::vector<hsize_t> offset;
    int64_t bytesWritten;
    uint64_t chunkCounter;
    std::string filePath;
    uint64_t rank;
    std::shared_ptr<holoscan::Tensor> permutedTensor;

    // Metrics.

    std::chrono::time_point<std::chrono::steady_clock> lastMeasurementTime;
    std::atomic<int64_t> bytesSinceLastMeasurement{0};
    std::atomic<double> currentBandwidthMBps{0.0};

    std::thread metricsThread;
    bool metricsThreadRunning;
    void metricsLoop();
};

void Hdf5SinkRdmaOp::initialize() {
    // Allocate memory.
    pimpl = new Impl();

    // Initialize operator.
    Operator::initialize();
}

Hdf5SinkRdmaOp::~Hdf5SinkRdmaOp() {
    delete pimpl;
}

void Hdf5SinkRdmaOp::setup(OperatorSpec& spec) {
    spec.input<DspBlock>("in")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));

    spec.param(filePath_, "file_path");
}

void Hdf5SinkRdmaOp::start() {
    // Convert Parameters to variables.

    pimpl->filePath = filePath_.get();

    // Initialize arrays.

    pimpl->rank = 3;  // Placeholder. Replace with actual dimensions.
    pimpl->dims.resize(pimpl->rank);
    pimpl->maxdims.resize(pimpl->rank);
    pimpl->chunkDims.resize(pimpl->rank);
    pimpl->count.resize(pimpl->rank);
    pimpl->offset.resize(pimpl->rank);

    // Set up HDF5 file.
    // TODO: Placeholder. Replace with actual dimensions.

    pimpl->dims = {0, 1, 192};
    pimpl->maxdims = {H5S_UNLIMITED, 1, 192};
    pimpl->chunkDims = {8192, 1, 192};
    pimpl->count = {1, 1, 192};
    pimpl->offset = {0, 0, 0};

    // Set up HDF5 library.

    pimpl->faplId = H5Pcreate(H5P_FILE_ACCESS);

    H5Pset_fapl_gds(pimpl->faplId, MBOUNDARY_DEF, FBSIZE_DEF, CBSIZE_DEF);

    // Create HDF5 file.

    if ((pimpl->fileId = H5Fcreate(pimpl->filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, pimpl->faplId)) < 0) {
        HOLOSCAN_LOG_ERROR("Error creating HDF5 file.");
        throw std::runtime_error("I/O Error");
    }

    // Create HDF5 dataspace with unlimited dimensions.
    pimpl->dataspaceId = H5Screate_simple(pimpl->rank, pimpl->dims.data(), pimpl->maxdims.data());

    // Create dataset creation property list and enable chunking.

    pimpl->plistId = H5Pcreate(H5P_DATASET_CREATE);

    HDF5_CHECK_THROW(H5Pset_chunk(pimpl->plistId, pimpl->rank, pimpl->chunkDims.data()), [&]{
        HOLOSCAN_LOG_ERROR("Error setting chunk size.");
    });

    // Create compound datatype for complex data.
    /*
    memtype_id = H5Tcreate(H5T_COMPOUND, sizeof(SensorData));
    H5Tinsert(memtype_id, "ID", HOFFSET(SensorData, id), H5T_NATIVE_INT);
    H5Tinsert(memtype_id, "Temperature", HOFFSET(SensorData, temperature), H5T_NATIVE_DOUBLE);
    H5Tinsert(memtype_id, "Location", HOFFSET(SensorData, location), H5T_NATIVE_CHAR);
    */
    // Create dataset.

    pimpl->datasetId = H5Dcreate2(pimpl->fileId,
                                  "data",
                                  H5T_IEEE_F64LE,
                                  pimpl->dataspaceId,
                                  H5P_DEFAULT,
                                  pimpl->plistId,
                                  H5P_DEFAULT);

    // Create memory dataspace.

    pimpl->memspaceId = H5Screate_simple(pimpl->rank, pimpl->chunkDims.data(), NULL);

    // TODO: Add HDF5 attributes.

    // Reset counters.

    pimpl->chunkCounter = 0;
    pimpl->bytesWritten = 0;
    pimpl->bytesSinceLastMeasurement = 0;
    pimpl->lastMeasurementTime = std::chrono::steady_clock::now();
    pimpl->currentBandwidthMBps = 0.0;

    // Start metrics thread.

    pimpl->metricsThreadRunning = true;
    pimpl->metricsThread = std::thread([&]{
        pimpl->metricsLoop();
    });
}

void Hdf5SinkRdmaOp::stop() {
    // Stop metrics thread.

    pimpl->metricsThreadRunning = false;
    if (pimpl->metricsThread.joinable()) {
        pimpl->metricsThread.join();
    }

    // Close HDF5.

    HDF5_CHECK_THROW(H5Sclose(pimpl->memspaceId), [&]{
        HOLOSCAN_LOG_ERROR("Error closing memory dataspace (memspaceId).");
    });
    HDF5_CHECK_THROW(H5Pclose(pimpl->plistId), [&]{
        HOLOSCAN_LOG_ERROR("Error closing memory dataspace (plistId).");
    });
    HDF5_CHECK_THROW(H5Dclose(pimpl->datasetId), [&]{
        HOLOSCAN_LOG_ERROR("Error closing memory dataspace (datasetId).");
    });
    HDF5_CHECK_THROW(H5Fclose(pimpl->fileId), [&]{
        HOLOSCAN_LOG_ERROR("Error closing memory dataspace (fileId).");
    });
}

void Hdf5SinkRdmaOp::compute(InputContext& input, OutputContext&, ExecutionContext&) {
    const auto& tensor = input.receive<DspBlock>("in").value().tensor;
    const auto& tensorBytes = tensor->size() * (tensor->dtype().bits / 8);

    // Allocate permuted tensor.

    if (pimpl->bytesWritten == 0) {
        CUDA_CHECK_THROW(DspBlockAlloc(tensor, pimpl->permutedTensor), [&]{
            HOLOSCAN_LOG_ERROR("Failed to allocate permuted tensor.");
        });

        GDS_CHECK_THROW(cuFileBufRegister(pimpl->permutedTensor->data(), tensorBytes, 0), [&]{
            HOLOSCAN_LOG_ERROR("Failed to register buffer with GDS driver.");
        });
    }

    // Permute tensor.

    CUDA_CHECK_THROW(DspBlockPermutation(pimpl->permutedTensor->to_dlpack(), tensor->to_dlpack()), [&]{
        HOLOSCAN_LOG_ERROR("Failed to permute tensor.");
    });

    // Write tensor to HDF5.

    pimpl->dims[0] += 8192;
    HDF5_CHECK_THROW(H5Dset_extent(pimpl->datasetId, pimpl->dims.data()), [&]{
        HOLOSCAN_LOG_ERROR("Error setting dataset extent.");
    });

    pimpl->dataspaceId = H5Dget_space(pimpl->datasetId);
    pimpl->offset[0] = pimpl->chunkCounter * 8192;

    HDF5_CHECK_THROW(H5Sselect_hyperslab(pimpl->dataspaceId,
                                         H5S_SELECT_SET,
                                         pimpl->offset.data(),
                                         NULL,
                                         pimpl->chunkDims.data(),
                                         NULL), [&]{
        HOLOSCAN_LOG_ERROR("Error selecting hyperslab.");
    });

    // Write all the data at once
    HDF5_CHECK_THROW(H5Dwrite(pimpl->datasetId,
                              H5T_IEEE_F64LE,
                              pimpl->memspaceId,
                              pimpl->dataspaceId,
                              H5P_DEFAULT,
                              pimpl->permutedTensor->data()), [&]{
        HOLOSCAN_LOG_ERROR("Error writing data to dataset.");
    });

    // TODO: Add HDF5 write.

    pimpl->chunkCounter += 1;
    pimpl->bytesWritten += tensorBytes;
    pimpl->bytesSinceLastMeasurement += tensorBytes;
}

void Hdf5SinkRdmaOp::Impl::metricsLoop() {
    while (metricsThreadRunning) {
        auto now = std::chrono::steady_clock::now();
        auto elapsedSeconds = std::chrono::duration<double>(now - lastMeasurementTime).count();

        if (elapsedSeconds > 0.0) {
            int64_t bytes = bytesSinceLastMeasurement.exchange(0);
            currentBandwidthMBps = static_cast<double>(bytes) / (1024.0 * 1024.0) / elapsedSeconds;
            lastMeasurementTime = now;
        }

        HOLOSCAN_LOG_INFO("HDF5 Sink RDMA Operator:");
        HOLOSCAN_LOG_INFO("  Current Bandwidth: {:.2f} MB/s", currentBandwidthMBps);
        HOLOSCAN_LOG_INFO("  Total Data Written: {:.0f} MB", static_cast<double>(bytesWritten) / (1024.0 * 1024.0));

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

}  // namespace stelline::operators::io
