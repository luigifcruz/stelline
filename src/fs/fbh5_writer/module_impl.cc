#include "module_impl.hh"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include <cufile.h>
#include <hdf5.h>

#include <jetstream/backend/devices/cuda/helpers.hh>
#include <jetstream/memory/axis.hh>

#include "../helpers.hh"
#include "../recording_axes.hh"

extern "C" {
#include "filterbankc99.h"
#include "h5dsc99/h5_dataspace.h"
}

#include "H5FDgds.h"

namespace Jetstream::Modules {

Result Fbh5WriterImpl::validate() {
    validatedExpectedShape = {};
    validatedInputDataType = DataType::None;

    if (!inputs().contains("input")) {
        return Result::SUCCESS;
    }

    const Tensor& input = inputs().at("input").tensor;
    if (!input.validShape() || input.size() == 0) {
        return Result::SUCCESS;
    }

    if (input.rank() != kExpectedRank) {
        JST_ERROR("[MODULE_FBH5_WRITER] Input tensor must have {} dimensions [T, B, C, I], but received shape {}.",
                  kExpectedRank,
                  input.shape());
        return Result::ERROR;
    }

    if (input.shape(kTimeAxis) > std::numeric_limits<std::size_t>::max() ||
        input.shape(kBeamAxis) > static_cast<U64>(std::numeric_limits<int>::max()) ||
        input.shape(kChannelAxis) > static_cast<U64>(std::numeric_limits<int>::max()) ||
        input.shape(kIfAxis) > static_cast<U64>(std::numeric_limits<int>::max())) {
        JST_ERROR("[MODULE_FBH5_WRITER] Input dimensions exceed the FBH5 format limits.");
        return Result::ERROR;
    }

    SignalAxes axes;
    if (ResolveSignalAxes(input, axes) != Result::SUCCESS ||
        axes.sample != Index{kTimeAxis} ||
        axes.channel != Index{kChannelAxis}) {
        JST_ERROR("[MODULE_FBH5_WRITER] Input tensor must define sampleAxis={} and channelAxis={} for [T, B, C, I].",
                  kTimeAxis,
                  kChannelAxis);
        return Result::ERROR;
    }

    validatedExpectedShape = input.shape();
    validatedInputDataType = input.dtype();

    return Result::SUCCESS;
}

Result Fbh5WriterImpl::define() {
    JST_CHECK(defineInterfaceInput("input"));

    return Result::SUCCESS;
}

Result Fbh5WriterImpl::create() {
    expectedShape = validatedExpectedShape;
    inputDataType = validatedInputDataType;
    openAttempted = false;
    timeline = {};
    blockWritten = false;

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
        JST_ERROR("[MODULE_FBH5_WRITER] File path is empty.");
        return Result::INCOMPLETE;
    }

    const auto parentPath = filePath.parent_path();
    if (!parentPath.empty()) {
        std::error_code ec;
        if (!std::filesystem::exists(parentPath, ec)) {
            JST_ERROR("[MODULE_FBH5_WRITER] Parent directory '{}' does not exist.", parentPath.string());
            return Result::INCOMPLETE;
        }
    }

    std::error_code fileQueryError;
    const bool fileExists = std::filesystem::exists(filePath, fileQueryError);
    if (fileQueryError) {
        JST_ERROR("[MODULE_FBH5_WRITER] Failed to query '{}'.", filepath);
        return Result::ERROR;
    }
    if (fileExists && !overwrite) {
        JST_ERROR("[MODULE_FBH5_WRITER] File '{}' already exists.", filepath);
        return Result::ERROR;
    }

    const U64 ntimesPerWrite = expectedShape[kTimeAxis];
    const U64 nbeams = expectedShape[kBeamAxis];
    const U64 nchans = expectedShape[kChannelAxis];
    const U64 nifs = expectedShape[kIfAxis];

    filterbank_header_t* header = &fbh5File.header;

    header->machine_id = 20;
    header->telescope_id = 6;
    header->data_type = 1;
    header->barycentric = 0;
    header->pulsarcentric = 0;
    header->src_raj = 20.0 + 39.0 / 60.0 + 7.4 / 3600.0;
    header->src_dej = 42.0 + 24.0 / 60.0 + 24.5 / 3600.0;
    header->az_start = 12.3456;
    header->za_start = 65.4321;
    header->nchans = static_cast<int>(nchans);
    header->nbeams = static_cast<int>(nbeams);
    header->ibeam = 1;
    header->nbits = inputDataType == DataType::F32 ? 32 : 0;
    header->nifs = static_cast<int>(nifs);

    std::strncpy(header->source_name, kSourceName, sizeof(header->source_name) - 1);
    header->source_name[sizeof(header->source_name) - 1] = '\0';

    fbh5File.nchans_per_write = static_cast<std::size_t>(nchans);
    fbh5File.ntimes_per_write = static_cast<std::size_t>(ntimesPerWrite);
    JST_CHECK(loadHeaderMetadata(inputs().at("input").tensor));

    // Keep both datasets unfiltered so GDS can write raw chunks.
    fbh5File.ds_data.filter_flag = H5_FILTER_FLAG_NONE;
    fbh5File.ds_mask.filter_flag = H5_FILTER_FLAG_NONE;

    faplId = H5Pcreate(H5P_FILE_ACCESS);
    if (faplId < 0) {
        JST_ERROR("[MODULE_FBH5_WRITER] Failed to create the HDF5 file access property list.");
        return Result::ERROR;
    }
    faplOpen = true;

    JST_HDF5_CHECK(H5Pset_fapl_gds(faplId, MBOUNDARY_DEF, FBSIZE_DEF, CBSIZE_DEF), [&] {
        JST_ERROR("[MODULE_FBH5_WRITER] Failed to configure the HDF5 GDS file access property list. Error {}.", err);
    });

    if (const auto error = ReservePartialRecording(filePath, partialPath)) {
        JST_ERROR("[MODULE_FBH5_WRITER] Failed to create a staging file beside '{}': {}.", filepath, error.message());
        return Result::ERROR;
    }

    hid_t dataTypeId = H5Tcopy(H5T_IEEE_F32LE);
    if (dataTypeId < 0) {
        JST_ERROR("[MODULE_FBH5_WRITER] Failed to create the FBH5 dataset data type.");
        return Result::ERROR;
    }

    const herr_t openStatus = filterbank_h5_open_explicit(partialPath.c_str(),
                                                          &fbh5File,
                                                          dataTypeId,
                                                          faplId);
    openAttempted = true;

    if (!fbh5File.ds_data.name) {
        JST_HDF5_CHECK(H5Tclose(dataTypeId), [&] {
            JST_ERROR("[MODULE_FBH5_WRITER] Failed to close the unused FBH5 dataset data type. Error {}.", err);
        });
    }

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

    bytesWritten.publish(0);

    return Result::SUCCESS;
}

Result Fbh5WriterImpl::loadHeaderMetadata(const Tensor& input) {
    if (!ReadBlockTiming(input)) {
        JST_ERROR("[MODULE_FBH5_WRITER] Recording requires U64 'timestamp' and positive U64 'duration' input attributes.");
        return Result::ERROR;
    }

    JST_CHECK(LoadRecordingSource(*environment(), "[MODULE_FBH5_WRITER]", source));

    filterbank_header_t& header = fbh5File.header;
    const auto frequencies = ChannelFrequencies(source.frequencyStart,
                                                source.frequencyStop,
                                                source.channelCount(),
                                                static_cast<U64>(header.nchans));
    if (!frequencies || !std::isfinite(frequencies->first) || !std::isfinite(frequencies->step)) {
        JST_ERROR("[MODULE_FBH5_WRITER] Input channel count {} does not split the {} F-engine channels of the instance band.",
                  header.nchans,
                  source.channelCount());
        return Result::ERROR;
    }

    header.fch1 = frequencies->first;
    header.foff = frequencies->step;

    return Result::SUCCESS;
}

Result Fbh5WriterImpl::planBlock(const Tensor& input, RecordingGap& gap) {
    JST_CHECK(timeline.plan(input, "[MODULE_FBH5_WRITER]", gap));

    JST_CHECK(CheckRecordingSource(*environment(), "[MODULE_FBH5_WRITER]", source));

    if (!blockWritten) {
        JST_CHECK(writeTiming(gap.timing));
    }
    return Result::SUCCESS;
}

Result Fbh5WriterImpl::commitBlock(const RecordingGap& gap) {
    timeline.commit(gap);
    blockWritten = true;
    if (partialPath.empty()) {
        return Result::SUCCESS;
    }
    JST_HDF5_CHECK(H5Fflush(fbh5File.file_id, H5F_SCOPE_GLOBAL), [&] {
        JST_ERROR("[MODULE_FBH5_WRITER] Failed to flush the new recording before publishing it. Error {}.", err);
    });
    if (const auto error = PublishPartialRecording(partialPath, filepath, overwrite)) {
        if (error == std::errc::file_exists) {
            JST_ERROR("[MODULE_FBH5_WRITER] File '{}' already exists.", filepath);
        } else {
            JST_ERROR("[MODULE_FBH5_WRITER] Failed to move the new recording into '{}'.", filepath);
        }
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

Result Fbh5WriterImpl::writeTiming(const BlockTiming& timing) {
    auto& header = fbh5File.header;
    header.tstart = UnixSecondsToMjd(RfSampleUnixSeconds(static_cast<U64>(source.synctime),
                                                         timing.timestamp,
                                                         source.samplePeriod));
    if (timing.duration % fbh5File.ntimes_per_write != 0) {
        JST_ERROR("[MODULE_FBH5_WRITER] Block duration of {} RF samples does not divide into {} spectra.",
                  timing.duration,
                  fbh5File.ntimes_per_write);
        return Result::ERROR;
    }
    const U64 samplesPerRow = timing.duration / fbh5File.ntimes_per_write;
    header.tsamp = static_cast<F64>(samplesPerRow) * source.samplePeriod;

    for (const auto& [name, value] : {std::pair{"tstart", &header.tstart}, std::pair{"tsamp", &header.tsamp}}) {
        const hid_t attribute = H5Aopen(fbh5File.ds_data.D_id, name, H5P_DEFAULT);
        if (attribute < 0) {
            JST_ERROR("[MODULE_FBH5_WRITER] Failed to open the FBH5 '{}' attribute.", name);
            return Result::ERROR;
        }
        const herr_t writeStatus = H5Awrite(attribute, H5T_NATIVE_DOUBLE, value);
        const herr_t closeStatus = H5Aclose(attribute);
        if (writeStatus < 0 || closeStatus < 0) {
            JST_ERROR("[MODULE_FBH5_WRITER] Failed to write the FBH5 '{}' attribute.", name);
            return Result::ERROR;
        }
    }
    const RecordingGrid grid{timing.timestamp, static_cast<U64>(source.synctime), samplesPerRow};
    JST_HDF5_CHECK(WriteRecordingGrid(fbh5File.ds_data.D_id, false, grid), [&] {
        JST_ERROR("[MODULE_FBH5_WRITER] Failed to write the FBH5 RF grid attributes. Error {}.", err);
    });
    return Result::SUCCESS;
}

Result Fbh5WriterImpl::appendChunk(const void* data) {
    JST_HDF5_CHECK(H5DSextend(&fbh5File.ds_data), [&] {
        JST_ERROR("[MODULE_FBH5_WRITER] Failed to extend the FBH5 data dataset. Error {}.", err);
    });
    JST_HDF5_CHECK(H5DSwrite(&fbh5File.ds_data, data), [&] {
        JST_ERROR("[MODULE_FBH5_WRITER] Failed to write the FBH5 data dataset. Error {}.", err);
    });
    JST_HDF5_CHECK(H5DSextend(&fbh5File.ds_mask), [&] {
        JST_ERROR("[MODULE_FBH5_WRITER] Failed to extend the FBH5 mask dataset. Error {}.", err);
    });
    JST_HDF5_CHECK(H5DSwrite(&fbh5File.ds_mask, mask), [&] {
        JST_ERROR("[MODULE_FBH5_WRITER] Failed to write the FBH5 mask dataset. Error {}.", err);
    });
    return Result::SUCCESS;
}

Result Fbh5WriterImpl::appendChunks(const void* data, const U64 samples, const U64 missingBlocks) {
    if (missingBlocks != 0) {
        std::vector<F32> missingData;
        try {
            missingData.resize(samples, 0.0f);
        } catch (const std::bad_alloc&) {
            JST_ERROR("[MODULE_FBH5_WRITER] Failed to allocate the missing-data buffer.");
            return Result::ERROR;
        }
        const auto maskBytes = H5DSsize(&fbh5File.ds_mask);
        std::memset(mask, 1, maskBytes);
        for (U64 block = 0; block < missingBlocks; ++block) {
            const Result result = appendChunk(missingData.data());
            if (result != Result::SUCCESS) {
                std::memset(mask, 0, maskBytes);
                return result;
            }
        }
        std::memset(mask, 0, maskBytes);
    }
    return appendChunk(data);
}

Result Fbh5WriterImpl::destroy() {
    Result result = Result::SUCCESS;

    if (bufferRegistered) {
        const CUfileError_t status =
            cuFileBufDeregister(const_cast<void*>(registeredBuffer));
        if (status.err != CU_FILE_SUCCESS) {
            JST_ERROR("[MODULE_FBH5_WRITER] Failed to deregister the previously registered input buffer (CUfile error {}).",
                      static_cast<int>(status.err));
            result = Result::ERROR;
        }
        bufferRegistered = false;
        registeredBuffer = nullptr;
        registeredBufferSize = 0;
    }

    if (mask) {
        std::free(mask);
        mask = nullptr;
        fbh5File.mask = nullptr;
    }

    if (openAttempted && fbh5File.ds_data.name) {
        const herr_t status = H5DSclose(&fbh5File.ds_data);
        if (status < 0) {
            JST_ERROR("[MODULE_FBH5_WRITER] Failed to close the FBH5 data dataset. Error {}.",
                      status);
            result = Result::ERROR;
        }
        dataOpen = false;
    }

    if (openAttempted && fbh5File.ds_mask.name) {
        const herr_t status = H5DSclose(&fbh5File.ds_mask);
        if (status < 0) {
            JST_ERROR("[MODULE_FBH5_WRITER] Failed to close the FBH5 mask dataset. Error {}.",
                      status);
            result = Result::ERROR;
        }
        maskOpen = false;
    }

    if (openAttempted && fbh5File.file_id > 0) {
        const herr_t status = H5Fclose(fbh5File.file_id);
        if (status < 0) {
            JST_ERROR("[MODULE_FBH5_WRITER] Failed to close the FBH5 file. Error {}.",
                      status);
            result = Result::ERROR;
        }
        fileOpen = false;
    }
    openAttempted = false;

    if (faplOpen) {
        const herr_t status = H5Pclose(faplId);
        if (status < 0) {
            JST_ERROR("[MODULE_FBH5_WRITER] Failed to close the HDF5 file access property list. Error {}.",
                      status);
            result = Result::ERROR;
        }
        faplOpen = false;
        faplId = -1;
    }

    RemovePartialRecording(partialPath);

    fbh5File = {};
    timeline = {};
    blockWritten = false;

    return result;
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
