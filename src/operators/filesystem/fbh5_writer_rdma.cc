#include <fcntl.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <hdf5.h>

#include <stelline/types.hh>
#include <stelline/operators/filesystem/base.hh>

#include "utils/helpers.hh"
#include "utils/modifiers.hh"

#include "H5FDgds.h"

extern "C" {
#include "h5dsc99/h5_dataspace.h"
}
#include "filterbankc99.h"

using namespace gxf;
using namespace holoscan;

namespace stelline::operators::filesystem {

struct Fbh5WriterRdmaOp::Impl {
    // State.

    std::string filePath;
    std::string scheme; // TODO make enum

    // FBHB file state.

    hid_t faplId;
    filterbank_h5_file_t fbh5_file;
    int64_t bytesWritten;
    uint64_t chunkCounter;
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

void Fbh5WriterRdmaOp::initialize() {
    // Allocate memory.
    pimpl = new Impl();

    // Initialize operator.
    Operator::initialize();
}

Fbh5WriterRdmaOp::~Fbh5WriterRdmaOp() {
    delete pimpl;
}

void Fbh5WriterRdmaOp::setup(OperatorSpec& spec) {
    spec.input<DspBlock>("in")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));

    spec.param(filePath_, "file_path");
}

void Fbh5WriterRdmaOp::start() {
    // Convert Parameters to variables.

    pimpl->filePath = filePath_.get();
    pimpl->scheme = "FBH5";

    // Initialize arrays.
    pimpl->fbh5_file = {0};
    pimpl->fbh5_file.header = {0};
    filterbank_header_t* hdr = &pimpl->fbh5_file.header;

    // 0=fake data; 1=Arecibo; 2=Ooty... others to be added
    hdr->machine_id = 20; // ???
    // 0=FAKE; 1=PSPM; 2=WAPP; 3=OOTY... others to be added
    hdr->telescope_id = 6; // GBT
    // 1=filterbank; 2=time series... others to be added
    hdr->data_type = 1;
    // 1 if barycentric or 0 otherwise (only output if non-zero)
    hdr->barycentric = 1;
    // 1 if pulsarcentric or 0 otherwise (only output if non-zero)
    hdr->pulsarcentric = 1;
    // right ascension (J2000) of source (hours)
    // will be converted to/from hhmmss.s
    hdr->src_raj = 20.0 + 39/60.0 + 7.4/3600.0;
    // declination (J2000) of source (degrees)
    // will be converted to/from ddmmss.s
    hdr->src_dej = 42.0 + 24/60.0 + 24.5/3600.0;
    // telescope azimuth at start of scan (degrees)
    hdr->az_start = 12.3456;
    // telescope zenith angle at start of scan (degrees)
    hdr->za_start = 65.4321;
    // centre frequency (MHz) of first filterbank channel
    hdr->fch1 = 4626.464842353016138;
    // filterbank channel bandwidth (MHz)
    hdr->foff = -0.000002793967724;
    // number of filterbank channels
    hdr->nchans = 192; // TODO: Placeholder. Replace with actual dimensions.
    // total number of beams
    hdr->nbeams = 1;
    // beam enumeration contained in this file
    hdr->ibeam = 1;
    // number of bits per time sample
    hdr->nbits = sizeof(float)*8;
    // time stamp (MJD) of first sample
    hdr->tstart = 57856.810798611114;
    // time interval between samples (s)
    hdr->tsamp = 1.825361100800;
    // number of seperate IF channels
    hdr->nifs = 1;
    // the name of the source being observed by the telescope
    // Max string size is supposed to be 80, but bug in sigproc if over 79
    strcpy(hdr->source_name, "1234567891123456789212345678931234567894123456789512345678961234567897123456789");

    pimpl->fbh5_file.nchans_per_write = hdr->nchans;
    pimpl->fbh5_file.ntimes_per_write = 8192; // TODO: placeholder. Replace with actual dimensions
    
    // Set up HDF5 library.

    pimpl->faplId = H5Pcreate(H5P_FILE_ACCESS);
    HDF5_CHECK_THROW(H5Pset_fapl_gds(pimpl->faplId, MBOUNDARY_DEF, FBSIZE_DEF, CBSIZE_DEF), [&]{
        HOLOSCAN_LOG_ERROR("Error setting the file access property list to H5FD_GDS.");
    });

    // Create HDF5 file.

    HDF5_CHECK_THROW(filterbank_h5_open_explicit(pimpl->filePath.data(), &pimpl->fbh5_file, H5Tcopy(H5T_IEEE_F64LE), pimpl->faplId), [&]{
        HOLOSCAN_LOG_ERROR("Error opening FBH5 file.");
    });

    // Create HDF5 data pointers

    pimpl->fbh5_file.mask = (uint8_t*) H5DSmalloc(&pimpl->fbh5_file.ds_mask);
    memset(pimpl->fbh5_file.mask, 0, H5DSsize(&pimpl->fbh5_file.ds_mask));

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

void Fbh5WriterRdmaOp::stop() {
    // Stop metrics thread.

    pimpl->metricsThreadRunning = false;
    if (pimpl->metricsThread.joinable()) {
        pimpl->metricsThread.join();
    }

    // Close HDF5.
    free(pimpl->fbh5_file.mask);
    HDF5_CHECK_THROW(H5DSclose(&pimpl->fbh5_file.ds_data), [&]{
        HOLOSCAN_LOG_ERROR("Error closing FBH5 data dataset (ds_data).");
    });
    HDF5_CHECK_THROW(H5DSclose(&pimpl->fbh5_file.ds_mask), [&]{
        HOLOSCAN_LOG_ERROR("Error closing FBH5 mask dataset (ds_mask).");
    });
    HDF5_CHECK_THROW(H5Fclose(pimpl->fbh5_file.file_id), [&]{
        HOLOSCAN_LOG_ERROR("Error closing FBH5 file.");
    });
}

void Fbh5WriterRdmaOp::compute(InputContext& input, OutputContext&, ExecutionContext&) {
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

    HDF5_CHECK_THROW(H5DSextend(&pimpl->fbh5_file.ds_data), [&]{
        HOLOSCAN_LOG_ERROR("H5DSextend failure on 'data'");
    });
    HDF5_CHECK_THROW(H5DSwrite(&pimpl->fbh5_file.ds_data, pimpl->permutedTensor->data()), [&]{
        HOLOSCAN_LOG_ERROR("H5DSwrite failure on 'data'");
    });

    // mask
    HDF5_CHECK_THROW(H5DSextend(&pimpl->fbh5_file.ds_mask), [&]{
        HOLOSCAN_LOG_ERROR("H5DSextend failure on 'mask'");
    });
    HDF5_CHECK_THROW(H5DSwrite(&pimpl->fbh5_file.ds_mask, pimpl->fbh5_file.mask), [&]{
        HOLOSCAN_LOG_ERROR("H5DSwrite failure on 'mask'");
    });

    // TODO: Add HDF5 write.

    pimpl->chunkCounter += 1;
    pimpl->bytesWritten += tensorBytes;
    pimpl->bytesSinceLastMeasurement += tensorBytes;
}

void Fbh5WriterRdmaOp::Impl::metricsLoop() {
    while (metricsThreadRunning) {
        auto now = std::chrono::steady_clock::now();
        auto elapsedSeconds = std::chrono::duration<double>(now - lastMeasurementTime).count();

        if (elapsedSeconds > 0.0) {
            int64_t bytes = bytesSinceLastMeasurement.exchange(0);
            currentBandwidthMBps = static_cast<double>(bytes) / (1024.0 * 1024.0) / elapsedSeconds;
            lastMeasurementTime = now;
        }

        HOLOSCAN_LOG_INFO("HDF5 Sink RDMA Operator:");
        HOLOSCAN_LOG_INFO("  Current Bandwidth: {:.2f} MB/s", currentBandwidthMBps.load());
        HOLOSCAN_LOG_INFO("  Total Data Written: {:.0f} MB", static_cast<double>(bytesWritten) / (1024.0 * 1024.0));

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

}  // namespace stelline::operators::io
