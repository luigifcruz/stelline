#include "module_impl.hh"

#include <cstdlib>
#include <cstring>
#include <filesystem>

#include <cufile.h>
#include <hdf5.h>

#include <jetstream/backend/devices/cuda/helpers.hh>
#include "jetstream/platform.hh"

#include "../helpers.hh"

extern "C" {
#include "filterbankc99.h"
#include "h5dsc99/h5_dataspace.h"
}

namespace Jetstream::Modules {

Result Fbh5ReaderImpl::validate() {
    return Result::SUCCESS;
}

Result Fbh5ReaderImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal"));
    JST_CHECK(defineInterfaceOutput("mask"));

    return Result::SUCCESS;
}

Result Fbh5ReaderImpl::create() {
    outputs()["signal"].produced(name(), "signal", buffer);
    outputs()["mask"].produced(name(), "mask", mask);

    batchCount.publish(0);
    currentBatchIndex.publish(0);
    currentBandwidth.publish(0.0f);
    bytesSinceLastMeasurement = 0;
    lastMeasurementTime = std::chrono::steady_clock::now();

    if (filepath.empty()) {
        JST_ERROR("[MODULE_FBH5_READER] File path is empty.");
        return Result::INCOMPLETE;
    }

    std::filesystem::path filePathNorm = std::filesystem::u8path(filepath);

    if (!std::filesystem::exists(filePathNorm)) {
        JST_ERROR("[MODULE_FBH5_READER] File '{}' does not exist.", filepath);
        return Result::INCOMPLETE;
    }

    
    // Suppress HDF5 automatic error printing — we handle errors manually.
    H5Eset_auto(H5E_DEFAULT, nullptr, nullptr);

    // Open the file read-only with the default (POSIX) VFD — no GDS required.
    fbh5File = filterbank_h5_access_file_explicit(
        filePathNorm.c_str(),
        faplId
    );
    if (fbh5File.file_id == H5I_INVALID_HID) {
        JST_ERROR("[MODULE_FBH5_READER] Cannot open file '{}'.", filepath);
        return Result::INCOMPLETE;
    }
    
    filterbank_h5_change_access_chunking(
        &fbh5File,
        batchSize, // nof time indices
        0, // shorthand for all IF
        0 // shorthand for all chans
    );
    
    JST_INFO("[MODULE_FBH5_READER] Opened '{}' — dim_chunks=[{}/{},{},{}].",
             filepath,
             batchSize,
             fbh5File.ds_data.dims[0], fbh5File.ds_data.dims[1], fbh5File.ds_data.dims[2]
             );
    batchCount.publish(fbh5File.ds_data.dims[0]/batchSize);

    DataType dataType;
    int nbits = H5Tget_size(fbh5File.ds_data.Tmem_id)*8; // distrust fbh5File.header.nbits
    switch (nbits) {
        case 32:
            dataType = DataType::F32;
            break;
        case 64:
            dataType = DataType::F64;
            break;
        default:
            JST_ERROR("[MODULE_FBH5_READER] Unsupported number of bits in '{}': {}.",
                      filepath,
                      nbits
                      );
    }
    JST_CHECK(buffer.create(device(), dataType, {fbh5File.ds_data.dimchunks[0], fbh5File.ds_data.dimchunks[1], fbh5File.ds_data.dims[2]}));
    if (buffer.sizeBytes() != H5DSsize(&fbh5File.ds_data)) {
        JST_ERROR("[MODULE_FBH5_READER] Signal buffer byte size is incorrect.");
        return Result::ERROR;
    }
    fbh5File.data = const_cast<void*>(buffer.data());

    JST_CHECK(mask.create(device(), DataType::U8, {fbh5File.ds_mask.dimchunks[0], fbh5File.ds_mask.dimchunks[1], fbh5File.ds_mask.dims[2]}));
    if (mask.sizeBytes() != H5DSsize(&fbh5File.ds_mask)) {
        JST_ERROR("[MODULE_FBH5_READER] Signal mask byte size is incorrect.");
        return Result::ERROR;
    }
    fbh5File.mask = (uint8_t*) mask.data();
    
    return Result::SUCCESS;
}

Result Fbh5ReaderImpl::destroy() {
    JST_INFO("[MODULE_FBH5_READER] destroy.");
    if (fbh5File.ds_data.D_id >= 0) {
        filterbank_h5_close(&fbh5File);
    }
    if (faplOpen) {
        JST_HDF5_CHECK(H5Pclose(faplId), [&] {
            JST_ERROR("[MODULE_FBH5_READER] Failed to close the HDF5 file access property list. Error {}.", err);
        });
        faplOpen = false;
        faplId = H5P_DEFAULT;
    }

    currentBandwidth.publish(0.0f);
    bytesSinceLastMeasurement = 0;

    fbh5File = {0};
    return Result::SUCCESS;
}

Result Fbh5ReaderImpl::reconfigure() {
    // the faplId is different between CPU and CUDA implementations
    return Result::RECREATE;
}

U64 Fbh5ReaderImpl::getCurrentBatchIndex() const {
    return currentBatchIndex.get();
}

U64 Fbh5ReaderImpl::getBatchCount() const {
    return batchCount.get();
}

F32 Fbh5ReaderImpl::getCurrentBandwidth() const {
    return currentBandwidth.get();
}

void Fbh5ReaderImpl::updateBandwidth(const U64 deltaBytes) {
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
