#include "module_impl.hh"

#include <exception>
#include <cstdlib>
#include <cstring>
#include <filesystem>

#include <hdf5.h>

#include <stelline/nexus.hh>

#include "../helpers.hh"

extern "C" {
#include "radiointerferometryc99.h"
}

#include "H5FDgds.h"

namespace Jetstream::Modules {

using stelline::Nexus;

namespace {

template<typename T>
Result RequireMetadata(const std::string& key, T& value) {
    if (Nexus::TryMetadataFetch(key, value)) {
        return Result::SUCCESS;
    }

    JST_WARN("[MODULE_UVH5_WRITER] Missing required Nexus metadata '{}'.", key);
    return Result::INCOMPLETE;
}

template<typename T>
Result RequireMetadata(const std::string& key, const U64 timestamp, T& value) {
    if (Nexus::TryMetadataFetch(key, value, timestamp)) {
        return Result::SUCCESS;
    }

    JST_WARN("[MODULE_UVH5_WRITER] Missing required Nexus metadata '{}' for timestamp {}.", key, timestamp);
    return Result::INCOMPLETE;
}

}  // namespace

Result Uvh5WriterImpl::validate() {
    const auto& config = *candidate();

    if (config.dspChannelizationRate == 0) {
        JST_ERROR("[MODULE_UVH5_WRITER] The 'dspChannelizationRate' must be positive.");
        return Result::ERROR;
    }

    if (config.dspIntegrationRate == 0) {
        JST_ERROR("[MODULE_UVH5_WRITER] The 'dspIntegrationRate' must be positive.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result Uvh5WriterImpl::define() {
    JST_CHECK(defineInterfaceInput("input"));

    return Result::SUCCESS;
}

Result Uvh5WriterImpl::loadMetadata() {
    metadata = {};

    JST_CHECK(RequireMetadata("observatory.name", metadata.telescopeName));
    JST_CHECK(RequireMetadata("observatory.coordinates.latitude", metadata.latitude));
    JST_CHECK(RequireMetadata("observatory.coordinates.longitude", metadata.longitude));
    JST_CHECK(RequireMetadata("observatory.coordinates.altitude", metadata.altitude));

    I32 antennaCount = 0;
    JST_CHECK(RequireMetadata("observation.antennas.length", antennaCount));
    if (antennaCount <= 0) {
        JST_WARN("[MODULE_UVH5_WRITER] Invalid observation antenna count {}.", antennaCount);
        return Result::INCOMPLETE;
    }

    metadata.antennas.resize(static_cast<size_t>(antennaCount));
    for (I32 index = 0; index < antennaCount; index++) {
        auto& antenna = metadata.antennas[static_cast<size_t>(index)];

        JST_CHECK(RequireMetadata(jst::fmt::format("observation.antennas.{}", index), antenna.name));
        JST_CHECK(RequireMetadata(jst::fmt::format("observatory.antenna.{}.number", antenna.name), antenna.number));
        JST_CHECK(RequireMetadata(jst::fmt::format("observatory.antenna.{}.diameter", antenna.name), antenna.diameter));
        JST_CHECK(RequireMetadata(jst::fmt::format("observatory.antenna.{}.position.x", antenna.name), antenna.position[0]));
        JST_CHECK(RequireMetadata(jst::fmt::format("observatory.antenna.{}.position.y", antenna.name), antenna.position[1]));
        JST_CHECK(RequireMetadata(jst::fmt::format("observatory.antenna.{}.position.z", antenna.name), antenna.position[2]));
    }

    const auto& firstAntenna = metadata.antennas.front().name;

    JST_CHECK(RequireMetadata(jst::fmt::format("observatory.antenna.{}.pointing.ra", firstAntenna), metadata.pointingRa));
    JST_CHECK(RequireMetadata(jst::fmt::format("observatory.antenna.{}.pointing.dec", firstAntenna), metadata.pointingDec));
    JST_CHECK(RequireMetadata(jst::fmt::format("observatory.antenna.{}.pointing.source_name", firstAntenna),
                              metadata.pointingSourceName));

    F64 frequencyStart = 0.0;
    F64 frequencyStop = 0.0;
    U64 channelStart = 0;
    U64 channelStop = 0;
    U64 bandCount = 0;

    JST_CHECK(RequireMetadata("instance.bands.len", bandCount));
    if (bandCount == 0) {
        JST_WARN("[MODULE_UVH5_WRITER] Invalid instance band count {}.", bandCount);
        return Result::INCOMPLETE;
    }

    // TODO: This is trash but rewrite needs Nexus Server change.

    const std::string bandKey = "instance.bands.0";

    JST_CHECK(RequireMetadata(jst::fmt::format("{}.tuning", bandKey), metadata.instanceTuning));
    JST_CHECK(RequireMetadata(jst::fmt::format("{}.band_index", bandKey), metadata.instanceBandIndex));
    JST_CHECK(RequireMetadata(jst::fmt::format("observatory.antenna.{}.tunings.{}.bands.{}.frequency_start",
                                               firstAntenna,
                                               metadata.instanceTuning,
                                               metadata.instanceBandIndex),
                              frequencyStart));
    JST_CHECK(RequireMetadata(jst::fmt::format("observatory.antenna.{}.tunings.{}.bands.{}.frequency_stop",
                                               firstAntenna,
                                               metadata.instanceTuning,
                                               metadata.instanceBandIndex),
                              frequencyStop));
    JST_CHECK(RequireMetadata(jst::fmt::format("observatory.antenna.{}.tunings.{}.bands.{}.channel_start",
                                               firstAntenna,
                                               metadata.instanceTuning,
                                               metadata.instanceBandIndex),
                              channelStart));
    JST_CHECK(RequireMetadata(jst::fmt::format("observatory.antenna.{}.tunings.{}.bands.{}.channel_stop",
                                               firstAntenna,
                                               metadata.instanceTuning,
                                               metadata.instanceBandIndex),
                              channelStop));

    metadata.instanceBandCenter = (frequencyStop + frequencyStart) / 2.0;
    metadata.instanceBandwidth = frequencyStop - frequencyStart;
    metadata.instanceChannelCount = channelStop - channelStart;
    if (metadata.instanceChannelCount == 0) {
        JST_WARN("[MODULE_UVH5_WRITER] Invalid instance channel count derived from '{}' and '{}'.",
                 channelStart,
                 channelStop);
        return Result::INCOMPLETE;
    }

    JST_CHECK(RequireMetadata(jst::fmt::format("observatory.antenna.{}.tunings.{}.fengine.synctime",
                                               firstAntenna,
                                               metadata.instanceTuning),
                              metadata.packetTimestampOffset));
    JST_CHECK(RequireMetadata(jst::fmt::format("observatory.antenna.{}.tunings.{}.fengine.sample_period",
                                               firstAntenna,
                                               metadata.instanceTuning),
                              metadata.sampleTimespan));

    JST_CHECK(RequireMetadata("observation.iers.pm_x_arcsec", metadata.iersPmXArcsec));
    JST_CHECK(RequireMetadata("observation.iers.pm_y_arcsec", metadata.iersPmYArcsec));
    JST_CHECK(RequireMetadata("observation.iers.ut1_utc", metadata.iersUt1Utc));

    integrationTimespan = metadata.sampleTimespan * static_cast<F64>(dspIntegrationRate);
    frequencyBandwidth = metadata.instanceBandwidth / static_cast<F64>(metadata.instanceChannelCount);
    frequencyBandwidth /= static_cast<F64>(dspChannelizationRate);

    metadata.valid = true;

    return Result::SUCCESS;
}

Result Uvh5WriterImpl::refreshDynamicMetadata(const U64& timestamp) {
    if (!metadata.valid || metadata.antennas.empty()) {
        return Result::INCOMPLETE;
    }

    const auto& firstAntenna = metadata.antennas.front().name;

    JST_CHECK(RequireMetadata(jst::fmt::format("observatory.antenna.{}.pointing.ra", firstAntenna),
                              timestamp,
                              metadata.pointingRa));
    JST_CHECK(RequireMetadata(jst::fmt::format("observatory.antenna.{}.pointing.dec", firstAntenna),
                              timestamp,
                              metadata.pointingDec));
    JST_CHECK(RequireMetadata(jst::fmt::format("observatory.antenna.{}.pointing.source_name", firstAntenna),
                              timestamp,
                              metadata.pointingSourceName));
    JST_CHECK(RequireMetadata("observation.iers.pm_x_arcsec", timestamp, metadata.iersPmXArcsec));
    JST_CHECK(RequireMetadata("observation.iers.pm_y_arcsec", timestamp, metadata.iersPmYArcsec));
    JST_CHECK(RequireMetadata("observation.iers.ut1_utc", timestamp, metadata.iersUt1Utc));

    return Result::SUCCESS;
}

Result Uvh5WriterImpl::create() {
    const Tensor& input = inputs().at("input").tensor;

    if (input.rank() != kExpectedRank) {
        JST_ERROR("[MODULE_UVH5_WRITER] Input tensor must have {} dimensions [T, BL, F, P], but received shape {}.",
                  kExpectedRank,
                  input.shape());
        return Result::ERROR;
    }

    if (!input.contiguous()) {
        JST_ERROR("[MODULE_UVH5_WRITER] Input tensor must be contiguous.");
        return Result::ERROR;
    }

    for (const auto& dim : input.shape()) {
        if (dim == 0) {
            JST_ERROR("[MODULE_UVH5_WRITER] Input tensor dimensions must be positive.");
            return Result::ERROR;
        }
    }

    if (input.shape(kTimeAxis) != kExpectedTimeCount) {
        JST_ERROR("[MODULE_UVH5_WRITER] Input tensor must have {} time sample per write, but received {}.",
                  kExpectedTimeCount,
                  input.shape(kTimeAxis));
        return Result::ERROR;
    }

    if (input.shape(kPolarizationAxis) != kExpectedPolarizationCount) {
        JST_ERROR("[MODULE_UVH5_WRITER] Input tensor must have {} polarizations, but received {}.",
                  kExpectedPolarizationCount,
                  input.shape(kPolarizationAxis));
        return Result::ERROR;
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
        JST_WARN("[MODULE_UVH5_WRITER] File path is empty.");
        return Result::INCOMPLETE;
    }

    const auto parentPath = filePath.parent_path();
    if (!parentPath.empty()) {
        std::error_code ec;
        if (!std::filesystem::exists(parentPath, ec)) {
            JST_WARN("[MODULE_UVH5_WRITER] Parent directory '{}' does not exist.", parentPath.string());
            return Result::INCOMPLETE;
        }
    }

    {
        std::error_code ec;
        const bool fileExists = std::filesystem::exists(filePath, ec);
        if (ec) {
            JST_ERROR("[MODULE_UVH5_WRITER] Failed to query '{}'.", filepath);
            return Result::ERROR;
        }

        if (fileExists) {
            if (!overwrite) {
                JST_ERROR("[MODULE_UVH5_WRITER] File '{}' already exists.", filepath);
                return Result::ERROR;
            }

            const bool removed = std::filesystem::remove(filePath, ec);
            if (ec || !removed) {
                JST_ERROR("[MODULE_UVH5_WRITER] Failed to remove '{}' before overwriting.", filepath);
                return Result::ERROR;
            }
        }
    }

    bytesWritten.publish(0);

    JST_CHECK(loadMetadata());

    const U64 expectedFrequencyCount = metadata.instanceChannelCount * dspChannelizationRate;
    if (expectedShape[kFrequencyAxis] != expectedFrequencyCount) {
        JST_ERROR("[MODULE_UVH5_WRITER] Input frequency dimension {} does not match the metadata-derived UVH5 frequency count {}.",
                  expectedShape[kFrequencyAxis],
                  expectedFrequencyCount);
        return Result::ERROR;
    }

    uvh5File = {};
    uvh5File.header = {};

    UVH5_header_t* header = &uvh5File.header;
    header->Ntimes = 0;
    header->Nblts = 0;
    header->Nfreqs = expectedFrequencyCount;
    header->Nspws = 1;

    const I32 antennaCount = static_cast<I32>(metadata.antennas.size());
    std::vector<UVH5_antinfo_t> antennaInfo(static_cast<size_t>(antennaCount));
    for (I32 index = 0; index < antennaCount; index++) {
        antennaInfo[static_cast<size_t>(index)].name = const_cast<char*>(metadata.antennas[static_cast<size_t>(index)].name.c_str());
        antennaInfo[static_cast<size_t>(index)].number = metadata.antennas[static_cast<size_t>(index)].number;
        antennaInfo[static_cast<size_t>(index)].position[0] = metadata.antennas[static_cast<size_t>(index)].position[0];
        antennaInfo[static_cast<size_t>(index)].position[1] = metadata.antennas[static_cast<size_t>(index)].position[1];
        antennaInfo[static_cast<size_t>(index)].position[2] = metadata.antennas[static_cast<size_t>(index)].position[2];
    }

    UVH5set_telescope_info(const_cast<char*>(metadata.telescopeName.c_str()),
                           metadata.latitude,
                           metadata.longitude,
                           metadata.altitude,
                           const_cast<char*>("xyz"),
                           metadata.antennas.front().diameter,
                           antennaCount,
                           antennaInfo.data(),
                           header);

    std::vector<char*> observationAntennas(static_cast<size_t>(antennaCount));
    for (I32 index = 0; index < antennaCount; index++) {
        observationAntennas[static_cast<size_t>(index)] = const_cast<char*>(metadata.antennas[static_cast<size_t>(index)].name.c_str());
    }

    UVH5set_observation_info(antennaCount,
                             observationAntennas.data(),
                             const_cast<char*>(kObservationPolarization),
                             header);

    if (expectedShape[kBaselineAxis] != static_cast<U64>(header->Nbls)) {
        JST_ERROR("[MODULE_UVH5_WRITER] Input baseline dimension {} does not match the UVH5 baseline count {}.",
                  expectedShape[kBaselineAxis],
                  header->Nbls);
        return Result::ERROR;
    }

    if (expectedShape[kPolarizationAxis] != static_cast<U64>(header->Npols)) {
        JST_ERROR("[MODULE_UVH5_WRITER] Input polarization dimension {} does not match the UVH5 polarization count {}.",
                  expectedShape[kPolarizationAxis],
                  header->Npols);
        return Result::ERROR;
    }

    I32 baselineIndex = 0;
    for (I32 antenna0 = 0; antenna0 < header->Nants_data; antenna0++) {
        const I32 antenna1Number = header->antenna_numbers[antenna0];
        for (I32 antenna1 = antenna0; antenna1 < header->Nants_data; antenna1++) {
            const I32 antenna2Number = header->antenna_numbers[antenna1];
            header->ant_1_array[baselineIndex] = antenna1Number;
            header->ant_2_array[baselineIndex] = antenna2Number;
            baselineIndex += 1;
        }
    }

    UVH5Hadmin(header);
    UVH5Hmalloc_phase_center_catalog(header, 1);

    header->phase_center_catalog[0].name = const_cast<char*>(metadata.pointingSourceName.c_str());
    header->phase_center_catalog[0].type = UVH5_PHASE_CENTER_SIDEREAL;
    header->phase_center_catalog[0].lon = calc_rad_from_degree(metadata.pointingRa * 360.0 / 24.0);
    header->phase_center_catalog[0].lat = calc_rad_from_degree(metadata.pointingDec);
    header->phase_center_catalog[0].frame = const_cast<char*>("icrs");
    header->phase_center_catalog[0].epoch = 2000.0;

    header->instrument = header->telescope_name;
    header->history = const_cast<char*>("None");

    if (header->phase_center_catalog[0].type == UVH5_PHASE_CENTER_DRIFTSCAN) {
        std::memcpy(header->_antenna_uvw_positions,
                    header->_antenna_enu_positions,
                    sizeof(double) * header->Nants_telescope * 3);
        UVH5permute_uvws(header);
    } else if (header->phase_center_catalog[0].type == UVH5_PHASE_CENTER_SIDEREAL) {
        std::memcpy(header->_antenna_uvw_positions,
                    header->_antenna_enu_positions,
                    sizeof(double) * header->Nants_telescope * 3);

        double hourAngleRad = 0.0;
        double declinationRad = 0.0;
        calc_position_to_uvw_frame_from_enu(header->_antenna_uvw_positions,
                                            header->Nants_data,
                                            hourAngleRad,
                                            declinationRad,
                                            calc_rad_from_degree(header->latitude));

        UVH5permute_uvws(header);
    }

    header->flex_spw = H5_FALSE;
    header->spw_array[0] = 1;

    const F64 instanceSubbandLower = metadata.instanceBandCenter - (metadata.instanceBandwidth / 2.0);
    header->channel_width[0] = frequencyBandwidth * 1e6;
    for (size_t index = 0; index < static_cast<size_t>(header->Nfreqs); index++) {
        header->channel_width[index] = header->channel_width[0];
        header->freq_array[index] = (instanceSubbandLower + (static_cast<F64>(index) + 0.5) * frequencyBandwidth) * 1e6;
    }

    faplId = H5Pcreate(H5P_FILE_ACCESS);
    if (faplId < 0) {
        JST_ERROR("[MODULE_UVH5_WRITER] Failed to create the HDF5 file access property list.");
        return Result::ERROR;
    }
    faplOpen = true;

    JST_HDF5_CHECK(H5Pset_fapl_gds(faplId, MBOUNDARY_DEF, FBSIZE_DEF, CBSIZE_DEF), [&] {
        JST_ERROR("[MODULE_UVH5_WRITER] Failed to configure the HDF5 GDS file access property list. Error {}.", err);
    });

    UVH5open_with_fileaccess(filepath.c_str(), &uvh5File, UVH5TcreateCF32(), faplId);
    if (H5Iis_valid(uvh5File.file_id) <= 0) {
        JST_ERROR("[MODULE_UVH5_WRITER] UVH5 open failed for '{}': invalid HDF5 file handle.", filepath);
        return Result::ERROR;
    }

    fileOpen = true;

    if (uvh5File.visdata) {
        std::free(uvh5File.visdata);
        uvh5File.visdata = nullptr;
    }

    return Result::SUCCESS;
}

Result Uvh5WriterImpl::destroy() {
    if (fileOpen) {
        uvh5File.visdata = std::malloc(8);
        if (!uvh5File.visdata) {
            JST_ERROR("[MODULE_UVH5_WRITER] Failed to allocate the UVH5 close placeholder buffer.");
            return Result::ERROR;
        }

        UVH5close(&uvh5File);
        uvh5File.visdata = nullptr;
        fileOpen = false;
    }

    if (faplOpen) {
        JST_HDF5_CHECK(H5Pclose(faplId), [&] {
            JST_ERROR("[MODULE_UVH5_WRITER] Failed to close the HDF5 file access property list. Error {}.", err);
        });
        faplOpen = false;
        faplId = -1;
    }

    uvh5File = {};

    return Result::SUCCESS;
}

Result Uvh5WriterImpl::reconfigure() {
    return Result::RECREATE;
}

void Uvh5WriterImpl::updateBandwidth(const U64 deltaBytes) {
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

F64 Uvh5WriterImpl::getBandwidthMBps() {
    return bandwidthMBps.get();
}

F64 Uvh5WriterImpl::getTotalDataWrittenMb() const {
    return static_cast<F64>(bytesWritten.get()) / (1024.0 * 1024.0);
}

U64 Uvh5WriterImpl::getChunkCounter() const {
    return chunkCounter.get();
}

}  // namespace Jetstream::Modules
