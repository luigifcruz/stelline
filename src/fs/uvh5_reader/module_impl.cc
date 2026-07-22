#include "module_impl.hh"

#include <cstdlib>
#include <cstring>
#include <filesystem>

#include <cufile.h>
#include <hdf5.h>

#include <jetstream/backend/devices/cuda/helpers.hh>

#include "../helpers.hh"

extern "C" {
#include "filterbankc99.h"
#include "h5dsc99/h5_dataspace.h"
}

namespace Jetstream::Modules {

Result Uvh5ReaderImpl::validate() {
    const auto& config = *candidate();
    if (config.batchSize == 0) {
        // TODO if zero, just read everything...
        JST_ERROR("[MODULE_UVH5_READER] The 'batchSize' must be positive.");
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

Result Uvh5ReaderImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result Uvh5ReaderImpl::create() {
    outputs()["signal"].produced(name(), "signal", buffer);
    batchCount.publish(0);
    currentBatchIndex.publish(0);
    currentBandwidth.publish(0.0f);
    bytesSinceLastMeasurement = 0;
    lastMeasurementTime = std::chrono::steady_clock::now();

    if (filepath.empty()) {
        JST_ERROR("[MODULE_UVH5_READER] File path is empty.");
        return Result::INCOMPLETE;
    }

    std::filesystem::path filePathNorm = std::filesystem::u8path(filepath);

    if (!std::filesystem::exists(filePathNorm)) {
        JST_ERROR("[MODULE_UVH5_READER] File '{}' does not exist.", filepath);
        return Result::INCOMPLETE;
    }

    
    // Suppress HDF5 automatic error printing — we handle errors manually.
    H5Eset_auto(H5E_DEFAULT, nullptr, nullptr);

    // Open the file read-only with the default (POSIX) VFD — no GDS required.
    uvh5File = UVH5access_file(
        filePathNorm.c_str(),
        faplId
    );
    if (uvh5File.file_id == H5I_INVALID_HID) {
        JST_ERROR("[MODULE_UVH5_READER] Cannot open file '{}'.", filepath);
        return Result::INCOMPLETE;
    }
    
    UVH5change_access_chunking(
        &uvh5File,
        batchSize // nof time-indices
    );
    JST_INFO("[MODULE_UVH5_READER] Opened '{}' — dim_chunks=[{}/{},{},{}].",
             filepath,
             batchSize*uvh5File.header.Nbls,
             uvh5File.DS_data_visdata.dims[0], uvh5File.DS_data_visdata.dims[1], uvh5File.DS_data_visdata.dims[2]
             );
    batchCount.publish(uvh5File.header.Ntimes/batchSize);

    DataType dataType;
    if (H5Tget_class(uvh5File.DS_data_visdata.Tmem_id) != H5T_COMPOUND || H5Tget_nmembers(uvh5File.DS_data_visdata.Tmem_id) != 2) {
        JST_ERROR("[MODULE_UVH5_READER] Visdata data type is not compound with 2 members.");
        return Result::ERROR;
    }
    int nbits = H5Tget_size(uvh5File.DS_data_visdata.Tmem_id)*8;
    switch (nbits) {
        case 64:
            dataType = DataType::CF32;
            break;
        case 128:
            dataType = DataType::CF64;
            break;
        default:
            JST_ERROR("[MODULE_UVH5_READER] Unsupported number of bits in '{}': {}.",
                      filepath,
                      nbits
                      );
            return Result::ERROR;
    }
    JST_CHECK(buffer.create(device(), dataType, {batchSize*uvh5File.header.Nbls, uvh5File.DS_data_visdata.dims[1], uvh5File.DS_data_visdata.dims[2]}));
    if (buffer.sizeBytes() != H5DSsize(&uvh5File.DS_data_visdata)) {
        JST_ERROR("[MODULE_UVH5_READER] Signal buffer size is incorrect. {} != {}", buffer.sizeBytes(), H5DSsize(&uvh5File.DS_data_visdata));
        return Result::ERROR;
    }
    uvh5File.visdata = buffer.data();
    uvh5File.flags = nullptr;
    uvh5File.nsamples = nullptr;

    return Result::SUCCESS;
}

Result Uvh5ReaderImpl::destroy() {
    if (uvh5File.DS_data_visdata.D_id >= 0) {
        uvh5File.visdata = nullptr;
        UVH5close(&uvh5File);
    }
    if (faplOpen) {
        JST_HDF5_CHECK(H5Pclose(faplId), [&] {
            JST_ERROR("[MODULE_UVH5_READER] Failed to close the HDF5 file access property list. Error {}.", err);
        });
        faplOpen = false;
        faplId = H5P_DEFAULT;
    }

    currentBandwidth.publish(0.0f);
    bytesSinceLastMeasurement = 0;

    uvh5File = {0};
    return Result::SUCCESS;
}

Result Uvh5ReaderImpl::reconfigure() {
    // the faplId is different between CPU and CUDA implementations
    return Result::RECREATE;
}

U64 Uvh5ReaderImpl::getCurrentBatchIndex() const {
    return currentBatchIndex.get();
}

U64 Uvh5ReaderImpl::getBatchCount() const {
    return batchCount.get();
}

F32 Uvh5ReaderImpl::getCurrentBandwidth() const {
    return currentBandwidth.get();
}

void Uvh5ReaderImpl::updateBandwidth(const U64 deltaBytes) {
    constexpr double kBandwidthMeasurementPeriodSeconds = 0.10;
    constexpr double kBandwidthEmaAlpha = 0.3;

    bytesSinceLastMeasurement += deltaBytes;

    const auto now = std::chrono::steady_clock::now();
    const double elapsedSeconds = std::chrono::duration<double>(now - lastMeasurementTime).count();
    if (elapsedSeconds < kBandwidthMeasurementPeriodSeconds) {
        return;
    }

    const double instantBandwidth = static_cast<double>(bytesSinceLastMeasurement) /
                                    static_cast<double>(JST_MB) /
                                    elapsedSeconds;
    const double smoothedBandwidth = kBandwidthEmaAlpha * instantBandwidth +
                                     (1.0 - kBandwidthEmaAlpha) * currentBandwidth.get();
    currentBandwidth.publish(static_cast<F32>(smoothedBandwidth));

    bytesSinceLastMeasurement = 0;
    lastMeasurementTime = now;
}

}  // namespace Jetstream::Modules
