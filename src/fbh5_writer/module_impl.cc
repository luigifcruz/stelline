#include "module_impl.hh"

#include <cstdlib>
#include <cstring>
#include <filesystem>

#include <cufile.h>
#include <hdf5.h>

#include <jetstream/backend/devices/cuda/helpers.hh>

#include "../net/helpers.hh"

extern "C" {
#include "filterbankc99.h"
#include "h5dsc99/h5_dataspace.h"
}

#include "H5FDgds.h"

namespace Jetstream::Modules {

Result Fbh5WriterImpl::validate() {
    return Result::SUCCESS;
}

Result Fbh5WriterImpl::define() {
    JST_CHECK(defineInterfaceInput("input"));

    return Result::SUCCESS;
}

Result Fbh5WriterImpl::create() {
    const Tensor& input = inputs().at("input").tensor;

    if (input.rank() != kExpectedRank) {
        JST_ERROR("[MODULE_FBH5_WRITER] Input tensor must have {} dimensions [T, B, C, I], but received shape {}.",
                  kExpectedRank,
                  input.shape());
        return Result::ERROR;
    }

    if (!input.contiguous()) {
        JST_ERROR("[MODULE_FBH5_WRITER] Input tensor must be contiguous.");
        return Result::ERROR;
    }

    for (const auto& dim : input.shape()) {
        if (dim == 0) {
            JST_ERROR("[MODULE_FBH5_WRITER] Input tensor dimensions must be positive.");
            return Result::ERROR;
        }
    }

    expectedShape = input.shape();
    inputDataType = input.dtype();

    bytesWritten.publish(0);
    chunkCounter.publish(0);
    bytesSinceLastMeasurement = 0;
    bandwidthMBps.publish(0.0);
    lastMeasurementTime = std::chrono::steady_clock::now();

    std::filesystem::path filePath;
    if (!filepath.empty()) {
        filePath = std::filesystem::path(filepath);

        std::error_code ec;
        if (std::filesystem::exists(filePath, ec) && !ec) {
            const U64 existingFileSize = std::filesystem::file_size(filePath, ec);
            if (!ec) {
                bytesWritten.publish(existingFileSize);
            }
        }
    }

    if (!recording) {
        return Result::SUCCESS;
    }

    if (filepath.empty()) {
        JST_WARN("[MODULE_FBH5_WRITER] File path is empty.");
        return Result::INCOMPLETE;
    }

    const auto parentPath = filePath.parent_path();
    if (!parentPath.empty()) {
        std::error_code ec;
        if (!std::filesystem::exists(parentPath, ec)) {
            JST_WARN("[MODULE_FBH5_WRITER] Parent directory '{}' does not exist.", parentPath.string());
            return Result::INCOMPLETE;
        }
    }

    {
        std::error_code ec;
        const bool fileExists = std::filesystem::exists(filePath, ec);
        if (ec) {
            JST_ERROR("[MODULE_FBH5_WRITER] Failed to query '{}'.", filepath);
            return Result::ERROR;
        }

        if (fileExists) {
            if (!overwrite) {
                JST_ERROR("[MODULE_FBH5_WRITER] File '{}' already exists.", filepath);
                return Result::ERROR;
            }

            const bool removed = std::filesystem::remove(filePath, ec);
            if (ec || !removed) {
                JST_ERROR("[MODULE_FBH5_WRITER] Failed to remove '{}' before overwriting.", filepath);
                return Result::ERROR;
            }
        }
    }

    bytesWritten.publish(0);

    const U64 ntimesPerWrite = expectedShape[kTimeAxis];
    const U64 nbeams = expectedShape[kBeamAxis];
    const U64 nchans = expectedShape[kChannelAxis];
    const U64 nifs = expectedShape[kIfAxis];

    filterbank_header_t* header = &fbh5File.header;

    header->machine_id = 20;
    header->telescope_id = 6;
    header->data_type = 1;
    header->barycentric = 1;
    header->pulsarcentric = 1;
    header->src_raj = 20.0 + 39.0 / 60.0 + 7.4 / 3600.0;
    header->src_dej = 42.0 + 24.0 / 60.0 + 24.5 / 3600.0;
    header->az_start = 12.3456;
    header->za_start = 65.4321;
    header->fch1 = 4626.464842353016138;
    header->foff = -0.000002793967724;
    header->nchans = nchans;
    header->nbeams = nbeams;
    header->ibeam = 1;
    header->nbits = inputDataType == DataType::F32 ? 32 : 0;
    header->tstart = 57856.810798611114;
    header->tsamp = 1.825361100800;
    header->nifs = nifs;

    std::strncpy(header->source_name, kSourceName, sizeof(header->source_name) - 1);
    header->source_name[sizeof(header->source_name) - 1] = '\0';

    fbh5File.nchans_per_write = nchans;
    fbh5File.ntimes_per_write = ntimesPerWrite;

    faplId = H5Pcreate(H5P_FILE_ACCESS);
    if (faplId < 0) {
        JST_ERROR("[MODULE_FBH5_WRITER] Failed to create the HDF5 file access property list.");
        return Result::ERROR;
    }
    faplOpen = true;

    JST_HDF5_CHECK(H5Pset_fapl_gds(faplId, MBOUNDARY_DEF, FBSIZE_DEF, CBSIZE_DEF), [&] {
        JST_ERROR("[MODULE_FBH5_WRITER] Failed to configure the HDF5 GDS file access property list. Error {}.", err);
    });

    hid_t dataTypeId = H5Tcopy(H5T_IEEE_F32LE);
    if (dataTypeId < 0) {
        JST_ERROR("[MODULE_FBH5_WRITER] Failed to create the FBH5 dataset data type.");
        return Result::ERROR;
    }

    const herr_t openStatus = filterbank_h5_open_explicit(filepath.c_str(),
                                                          &fbh5File,
                                                          dataTypeId,
                                                          faplId);

    JST_HDF5_CHECK(H5Tclose(dataTypeId), [&] {
        JST_ERROR("[MODULE_FBH5_WRITER] Failed to close the FBH5 dataset data type. Error {}.", err);
    });

    JST_HDF5_CHECK(openStatus, [&] {
        JST_ERROR("[MODULE_FBH5_WRITER] Failed to open the FBH5 file. Error {}.", err);
    });

    fileOpen = true;
    dataOpen = true;
    maskOpen = true;

    mask = static_cast<uint8_t*>(H5DSmalloc(&fbh5File.ds_mask));
    if (!mask) {
        JST_ERROR("[MODULE_FBH5_WRITER] Failed to allocate the FBH5 mask dataset buffer.");
        return Result::ERROR;
    }

    fbh5File.mask = mask;
    std::memset(mask, 0, H5DSsize(&fbh5File.ds_mask));

    return Result::SUCCESS;
}

Result Fbh5WriterImpl::destroy() {
    if (bufferRegistered) {
        JST_CUFILE_CHECK(cuFileBufDeregister(const_cast<void*>(registeredBuffer)), [&] {
            JST_ERROR("[MODULE_FBH5_WRITER] Failed to deregister the previously registered input buffer (CUfile error {}).",
                      err);
        });
        bufferRegistered = false;
        registeredBuffer = nullptr;
        registeredBufferSize = 0;
    }

    if (mask) {
        std::free(mask);
        mask = nullptr;
        fbh5File.mask = nullptr;
    }

    if (dataOpen) {
        JST_HDF5_CHECK(H5DSclose(&fbh5File.ds_data), [&] {
            JST_ERROR("[MODULE_FBH5_WRITER] Failed to close the FBH5 data dataset. Error {}.", err);
        });
        dataOpen = false;
    }

    if (maskOpen) {
        JST_HDF5_CHECK(H5DSclose(&fbh5File.ds_mask), [&] {
            JST_ERROR("[MODULE_FBH5_WRITER] Failed to close the FBH5 mask dataset. Error {}.", err);
        });
        maskOpen = false;
    }

    if (fileOpen) {
        JST_HDF5_CHECK(H5Fclose(fbh5File.file_id), [&] {
            JST_ERROR("[MODULE_FBH5_WRITER] Failed to close the FBH5 file. Error {}.", err);
        });
        fileOpen = false;
    }

    if (faplOpen) {
        JST_HDF5_CHECK(H5Pclose(faplId), [&] {
            JST_ERROR("[MODULE_FBH5_WRITER] Failed to close the HDF5 file access property list. Error {}.", err);
        });
        faplOpen = false;
        faplId = -1;
    }

    fbh5File = {};

    return Result::SUCCESS;
}

Result Fbh5WriterImpl::reconfigure() {
    return Result::RECREATE;
}

void Fbh5WriterImpl::updateBandwidth(const U64 deltaBytes) {
    constexpr double kBandwidthMeasurementPeriodSeconds = 0.10;
    constexpr double kBandwidthEmaAlpha = 0.3;

    bytesSinceLastMeasurement += static_cast<I64>(deltaBytes);

    const auto now = std::chrono::steady_clock::now();
    const F64 elapsedSeconds = std::chrono::duration<F64>(now - lastMeasurementTime).count();
    if (elapsedSeconds >= kBandwidthMeasurementPeriodSeconds) {
        const F64 instantMBps = elapsedSeconds > 0.0
                                    ? static_cast<F64>(bytesSinceLastMeasurement) / (1024.0 * 1024.0) / elapsedSeconds
                                    : 0.0;

        bandwidthMBps.publish(kBandwidthEmaAlpha * instantMBps +
                              (1.0 - kBandwidthEmaAlpha) * bandwidthMBps.get());

        bytesSinceLastMeasurement = 0;
        lastMeasurementTime = now;
    }
}

F64 Fbh5WriterImpl::getBandwidthMBps() {
    return bandwidthMBps.get();
}

F64 Fbh5WriterImpl::getTotalDataWrittenMb() const {
    return static_cast<F64>(bytesWritten.get()) / (1024.0 * 1024.0);
}

U64 Fbh5WriterImpl::getChunkCounter() const {
    return chunkCounter.get();
}

}  // namespace Jetstream::Modules
