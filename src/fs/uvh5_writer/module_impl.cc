#include "module_impl.hh"

#include <algorithm>
#include <cmath>
#include <exception>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>

#include <hdf5.h>

#include <jetstream/memory/axis.hh>
#include <jetstream/tools/numeric.hh>

#include "../helpers.hh"
#include "../recording_axes.hh"

extern "C" {
#include "radiointerferometryc99.h"
}

#include "H5FDgds.h"

namespace Jetstream::Modules {

Result Uvh5WriterImpl::validate() {
    validatedExpectedShape = {};
    validatedInputDataType = DataType::None;

    const auto& config = *candidate();

    if (!inputs().contains("input")) {
        return Result::SUCCESS;
    }

    const Tensor& input = inputs().at("input").tensor;
    if (!input.validShape() || input.size() == 0) {
        return Result::SUCCESS;
    }

    if (input.rank() != kExpectedRank) {
        JST_ERROR("[MODULE_UVH5_WRITER] Input tensor must have {} dimensions [T, BL, F, P], but received shape {}.",
                  kExpectedRank,
                  input.shape());
        return Result::ERROR;
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

    SignalAxes axes;
    if (ResolveSignalAxes(input, axes) != Result::SUCCESS ||
        axes.sample != Index{kTimeAxis} ||
        axes.channel != Index{kFrequencyAxis}) {
        JST_ERROR("[MODULE_UVH5_WRITER] Input tensor must define sampleAxis={} and channelAxis={} for [T, BL, F, P].",
                  kTimeAxis,
                  kFrequencyAxis);
        return Result::ERROR;
    }

    if (config.recording && !ReadBlockTiming(input)) {
        JST_ERROR("[MODULE_UVH5_WRITER] Recording requires U64 'timestamp' and positive U64 'duration' input attributes.");
        return Result::ERROR;
    }

    validatedExpectedShape = input.shape();
    validatedInputDataType = input.dtype();

    return Result::SUCCESS;
}

Result Uvh5WriterImpl::define() {
    JST_CHECK(defineInterfaceInput("input"));

    return Result::SUCCESS;
}

Result Uvh5WriterImpl::loadMetadata() {
    metadata = {};

    struct {
        std::string value;
        JST_SERDES(value);
    } text;

    struct {
        F64 value = 0.0;
        JST_SERDES(value);
    } real;

    struct {
        I64 value = 0;
        JST_SERDES(value);
    } integer;

    if (!environment()->tryGet("observatory.name", text)) {
        JST_ERROR("[MODULE_UVH5_WRITER] Missing required 'observatory.name' environment value.");
        return Result::INCOMPLETE;
    }
    metadata.telescopeName = text.value;

    if (!environment()->tryGet("observatory.coordinates.latitude", real)) {
        JST_ERROR("[MODULE_UVH5_WRITER] Missing required 'observatory.coordinates.latitude' environment value.");
        return Result::INCOMPLETE;
    }
    metadata.latitude = real.value;

    if (!environment()->tryGet("observatory.coordinates.longitude", real)) {
        JST_ERROR("[MODULE_UVH5_WRITER] Missing required 'observatory.coordinates.longitude' environment value.");
        return Result::INCOMPLETE;
    }
    metadata.longitude = real.value;

    if (!environment()->tryGet("observatory.coordinates.altitude", real)) {
        JST_ERROR("[MODULE_UVH5_WRITER] Missing required 'observatory.coordinates.altitude' environment value.");
        return Result::INCOMPLETE;
    }
    metadata.altitude = real.value;

    if (!environment()->tryGet("observation.antennas.length", integer)) {
        JST_ERROR("[MODULE_UVH5_WRITER] Missing required 'observation.antennas.length' environment value.");
        return Result::INCOMPLETE;
    }

    if (integer.value <= 0 ||
        integer.value > std::numeric_limits<I32>::max()) {
        JST_ERROR("[MODULE_UVH5_WRITER] Invalid observation antenna count {}.", integer.value);
        return Result::INCOMPLETE;
    }

    metadata.antennas.resize(static_cast<size_t>(integer.value));
    for (size_t index = 0; index < metadata.antennas.size(); index++) {
        auto& antenna = metadata.antennas[index];

        {
            const auto key = jst::fmt::format("observation.antennas.{}", index);
            if (!environment()->tryGet(key, text)) {
                JST_ERROR("[MODULE_UVH5_WRITER] Missing required '{}' environment value.", key);
                return Result::INCOMPLETE;
            }
            antenna.name = text.value;
        }

        {
            const auto key = jst::fmt::format("observatory.antenna.{}.number", antenna.name);
            if (!environment()->tryGet(key, integer)) {
                JST_ERROR("[MODULE_UVH5_WRITER] Missing required '{}' environment value.", key);
                return Result::INCOMPLETE;
            }
            if (integer.value < 0 ||
                static_cast<U64>(integer.value) > static_cast<U64>(std::numeric_limits<I32>::max())) {
                JST_ERROR("[MODULE_UVH5_WRITER] Invalid antenna number {} for '{}'.", integer.value, key);
                return Result::ERROR;
            }
            antenna.number = static_cast<U64>(integer.value);
        }

        {
            const auto key = jst::fmt::format("observatory.antenna.{}.diameter", antenna.name);
            if (!environment()->tryGet(key, real)) {
                JST_ERROR("[MODULE_UVH5_WRITER] Missing required '{}' environment value.", key);
                return Result::INCOMPLETE;
            }
            antenna.diameter = real.value;
        }

        for (size_t axis = 0; axis < antenna.position.size(); axis++) {
            const auto key = jst::fmt::format("observatory.antenna.{}.position.{}", antenna.name, "xyz"[axis]);
            if (!environment()->tryGet(key, real)) {
                JST_ERROR("[MODULE_UVH5_WRITER] Missing required '{}' environment value.", key);
                return Result::INCOMPLETE;
            }
            antenna.position[axis] = real.value;
        }
    }

    const auto& firstAntenna = metadata.antennas.front().name;

    {
        const auto key = jst::fmt::format("observatory.antenna.{}.pointing.ra", firstAntenna);
        if (!environment()->tryGet(key, real)) {
            JST_ERROR("[MODULE_UVH5_WRITER] Missing required '{}' environment value.", key);
            return Result::INCOMPLETE;
        }
        metadata.pointingRa = real.value;
    }

    {
        const auto key = jst::fmt::format("observatory.antenna.{}.pointing.dec", firstAntenna);
        if (!environment()->tryGet(key, real)) {
            JST_ERROR("[MODULE_UVH5_WRITER] Missing required '{}' environment value.", key);
            return Result::INCOMPLETE;
        }
        metadata.pointingDec = real.value;
    }

    {
        const auto key = jst::fmt::format("observatory.antenna.{}.pointing.source_name", firstAntenna);
        if (!environment()->tryGet(key, text)) {
            JST_ERROR("[MODULE_UVH5_WRITER] Missing required '{}' environment value.", key);
            return Result::INCOMPLETE;
        }
        metadata.pointingSourceName = text.value;
    }

    JST_CHECK(LoadRecordingSource(*environment(), "[MODULE_UVH5_WRITER]", source));
    metadata.instanceBandCenter = (source.frequencyStop + source.frequencyStart) / 2.0;
    metadata.instanceBandwidth = source.frequencyStop - source.frequencyStart;
    metadata.instanceChannelCount = source.channelCount();
    metadata.packetTimestampOffset = static_cast<U64>(source.synctime);
    metadata.sampleTimespan = source.samplePeriod;

    if (!environment()->tryGet("observation.iers.pm_x_arcsec", real)) {
        JST_ERROR("[MODULE_UVH5_WRITER] Missing required 'observation.iers.pm_x_arcsec' environment value.");
        return Result::INCOMPLETE;
    }
    metadata.iersPmXArcsec = real.value;

    if (!environment()->tryGet("observation.iers.pm_y_arcsec", real)) {
        JST_ERROR("[MODULE_UVH5_WRITER] Missing required 'observation.iers.pm_y_arcsec' environment value.");
        return Result::INCOMPLETE;
    }
    metadata.iersPmYArcsec = real.value;

    if (!environment()->tryGet("observation.iers.ut1_utc", real)) {
        JST_ERROR("[MODULE_UVH5_WRITER] Missing required 'observation.iers.ut1_utc' environment value.");
        return Result::INCOMPLETE;
    }
    metadata.iersUt1Utc = real.value;

    metadata.valid = true;

    return Result::SUCCESS;
}

Result Uvh5WriterImpl::prepareIntegration(const U64 timestamp,
                                         const F64 durationSeconds,
                                         const bool missing) {
    constexpr long double kUnixEpochJd = 2440587.5L;
    const long double midpoint = static_cast<long double>(metadata.packetTimestampOffset) +
                                 static_cast<long double>(timestamp) * metadata.sampleTimespan +
                                 durationSeconds / 2.0L;
    const auto jd = static_cast<F64>(kUnixEpochJd + midpoint / RADIOINTERFEROMETERY_DAYSEC);
    if (!std::isfinite(jd) || !std::isfinite(durationSeconds) || durationSeconds <= 0.0) {
        JST_ERROR("[MODULE_UVH5_WRITER] Invalid UVH5 integration time.");
        return Result::ERROR;
    }

    auto& header = uvh5File.header;
    const U64 samples = static_cast<U64>(header.Nbls) * header.Nfreqs * header.Npols;
    std::fill_n(header.time_array, header.Nbls, jd);
    std::fill_n(header.integration_time, header.Nbls, durationSeconds);
    std::fill_n(uvh5File.nsamples, samples, missing ? 0.0 : static_cast<F64>(samples));
    std::fill_n(uvh5File.flags, samples, missing ? H5_TRUE : H5_FALSE);
    return Result::SUCCESS;
}

Result Uvh5WriterImpl::refreshDynamicMetadata(const U64& timestamp) {
    if (!metadata.valid || metadata.antennas.empty()) {
        JST_ERROR("[MODULE_UVH5_WRITER] UVH5 metadata is not initialized.");
        return Result::INCOMPLETE;
    }

    const auto& firstAntenna = metadata.antennas.front().name;

    struct {
        F64 value = 0.0;
        JST_SERDES(value);
    } real;

    struct {
        std::string value;
        JST_SERDES(value);
    } text;

    {
        const auto key = jst::fmt::format("observatory.antenna.{}.pointing.ra", firstAntenna);
        if (!environment()->tryGet(key, real, timestamp)) {
            JST_ERROR("[MODULE_UVH5_WRITER] Missing environment value '{}' for timestamp {}.", key, timestamp);
            return Result::INCOMPLETE;
        }
        metadata.pointingRa = real.value;
    }

    {
        const auto key = jst::fmt::format("observatory.antenna.{}.pointing.dec", firstAntenna);
        if (!environment()->tryGet(key, real, timestamp)) {
            JST_ERROR("[MODULE_UVH5_WRITER] Missing environment value '{}' for timestamp {}.", key, timestamp);
            return Result::INCOMPLETE;
        }
        metadata.pointingDec = real.value;
    }

    {
        const auto key = jst::fmt::format("observatory.antenna.{}.pointing.source_name", firstAntenna);
        if (!environment()->tryGet(key, text, timestamp)) {
            JST_ERROR("[MODULE_UVH5_WRITER] Missing environment value '{}' for timestamp {}.", key, timestamp);
            return Result::INCOMPLETE;
        }
        metadata.pointingSourceName = text.value;
    }

    if (!environment()->tryGet("observation.iers.pm_x_arcsec", real, timestamp)) {
        JST_ERROR("[MODULE_UVH5_WRITER] Missing environment value 'observation.iers.pm_x_arcsec' for timestamp {}.", timestamp);
        return Result::INCOMPLETE;
    }
    metadata.iersPmXArcsec = real.value;

    if (!environment()->tryGet("observation.iers.pm_y_arcsec", real, timestamp)) {
        JST_ERROR("[MODULE_UVH5_WRITER] Missing environment value 'observation.iers.pm_y_arcsec' for timestamp {}.", timestamp);
        return Result::INCOMPLETE;
    }
    metadata.iersPmYArcsec = real.value;

    if (!environment()->tryGet("observation.iers.ut1_utc", real, timestamp)) {
        JST_ERROR("[MODULE_UVH5_WRITER] Missing environment value 'observation.iers.ut1_utc' for timestamp {}.", timestamp);
        return Result::INCOMPLETE;
    }
    metadata.iersUt1Utc = real.value;

    return Result::SUCCESS;
}

Result Uvh5WriterImpl::create() {
    expectedShape = validatedExpectedShape;
    inputDataType = validatedInputDataType;
    openAttempted = false;
    timeline = {};

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
        JST_ERROR("[MODULE_UVH5_WRITER] File path is empty.");
        return Result::INCOMPLETE;
    }

    const auto parentPath = filePath.parent_path();
    if (!parentPath.empty()) {
        std::error_code ec;
        if (!std::filesystem::exists(parentPath, ec)) {
            JST_ERROR("[MODULE_UVH5_WRITER] Parent directory '{}' does not exist.", parentPath.string());
            return Result::INCOMPLETE;
        }
    }

    std::error_code fileQueryError;
    const bool fileExists = std::filesystem::exists(filePath, fileQueryError);
    if (fileQueryError) {
        JST_ERROR("[MODULE_UVH5_WRITER] Failed to query '{}'.", filepath);
        return Result::ERROR;
    }
    if (fileExists && !overwrite) {
        JST_ERROR("[MODULE_UVH5_WRITER] File '{}' already exists.", filepath);
        return Result::ERROR;
    }

    JST_CHECK(loadMetadata());

    const U64 expectedFrequencyCount = expectedShape[kFrequencyAxis];
    if (expectedFrequencyCount > static_cast<U64>(std::numeric_limits<I32>::max())) {
        JST_ERROR("[MODULE_UVH5_WRITER] Input frequency dimension {} is too large for UVH5.", expectedFrequencyCount);
        return Result::ERROR;
    }
    if (expectedFrequencyCount % metadata.instanceChannelCount != 0) {
        JST_ERROR("[MODULE_UVH5_WRITER] Input channel count {} does not split the {} F-engine channels of the instance band.",
                  expectedFrequencyCount,
                  metadata.instanceChannelCount);
        return Result::ERROR;
    }

    const U64 antennaCountU64 = metadata.antennas.size();
    U64 antennaProduct = 0;
    if (!detail::CheckedMultiply(antennaCountU64,
                                 antennaCountU64 + 1,
                                 antennaProduct) ||
        antennaProduct > static_cast<U64>(std::numeric_limits<I32>::max())) {
        JST_ERROR("[MODULE_UVH5_WRITER] Observation antenna count is too large for UVH5.");
        return Result::ERROR;
    }
    const U64 expectedBaselineCount = antennaProduct / 2;
    if (expectedShape[kBaselineAxis] != expectedBaselineCount) {
        JST_ERROR("[MODULE_UVH5_WRITER] Input baseline dimension {} does not match the UVH5 baseline count {}.",
                  expectedShape[kBaselineAxis],
                  expectedBaselineCount);
        return Result::ERROR;
    }

    uvh5File = {};
    uvh5File.header = {};

    UVH5_header_t* header = &uvh5File.header;
    header->Ntimes = 0;
    header->Nblts = 0;
    header->Nfreqs = static_cast<I32>(expectedFrequencyCount);
    header->Nspws = 1;

    const I32 antennaCount = static_cast<I32>(antennaCountU64);
    std::vector<UVH5_antinfo_t> antennaInfo(static_cast<size_t>(antennaCount));
    for (I32 index = 0; index < antennaCount; index++) {
        antennaInfo[static_cast<size_t>(index)].name = const_cast<char*>(metadata.antennas[static_cast<size_t>(index)].name.c_str());
        antennaInfo[static_cast<size_t>(index)].number =
            static_cast<I32>(metadata.antennas[static_cast<size_t>(index)].number);
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
    headerAllocated = true;

    std::vector<char*> observationAntennas(static_cast<size_t>(antennaCount));
    for (I32 index = 0; index < antennaCount; index++) {
        observationAntennas[static_cast<size_t>(index)] = const_cast<char*>(metadata.antennas[static_cast<size_t>(index)].name.c_str());
    }

    UVH5set_observation_info(antennaCount,
                             observationAntennas.data(),
                             const_cast<char*>(kObservationPolarization),
                             header);

    if (header->Nbls < 0 ||
        static_cast<U64>(header->Nbls) != expectedBaselineCount) {
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

    const auto frequencies = ChannelFrequencies(metadata.instanceBandCenter - metadata.instanceBandwidth / 2.0,
                                                metadata.instanceBandCenter + metadata.instanceBandwidth / 2.0,
                                                metadata.instanceChannelCount,
                                                expectedFrequencyCount);
    if (!frequencies) {
        JST_ERROR("[MODULE_UVH5_WRITER] Invalid instance band frequency range.");
        return Result::ERROR;
    }
    for (size_t index = 0; index < static_cast<size_t>(header->Nfreqs); index++) {
        header->channel_width[index] = frequencies->step * 1e6;
        header->freq_array[index] = (frequencies->first + static_cast<F64>(index) * frequencies->step) * 1e6;
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

    if (const auto error = ReservePartialRecording(filePath, partialPath)) {
        JST_ERROR("[MODULE_UVH5_WRITER] Failed to create a staging file beside '{}': {}.", filepath, error.message());
        return Result::ERROR;
    }

    const hid_t dataTypeId = UVH5TcreateCF32();
    if (dataTypeId < 0) {
        JST_ERROR("[MODULE_UVH5_WRITER] Failed to create the UVH5 visibility data type.");
        return Result::ERROR;
    }

    // Keep visibility and nsamples data unfiltered so GDS can write GPU buffers directly.
    uvh5File.DS_data_visdata.filter_flag = H5_FILTER_FLAG_NONE;
    uvh5File.DS_data_nsamples.filter_flag = H5_FILTER_FLAG_NONE;
    // TODO: Evaluate flags compression overhead, file size, and cached compressed-chunk reuse.

    UVH5open_with_fileaccess(partialPath.c_str(), &uvh5File, dataTypeId, faplId);
    openAttempted = true;

    if (H5Iis_valid(uvh5File.file_id) <= 0 ||
        H5Iis_valid(uvh5File.header_id) <= 0 ||
        H5Iis_valid(uvh5File.data_id) <= 0 ||
        H5Iis_valid(uvh5File.keywords_id) <= 0 ||
        H5Iis_valid(uvh5File.DS_header_Ntimes.D_id) <= 0 ||
        H5Iis_valid(uvh5File.DS_header_Nblts.D_id) <= 0 ||
        H5Iis_valid(uvh5File.DS_header_ant_1_array.D_id) <= 0 ||
        H5Iis_valid(uvh5File.DS_header_ant_2_array.D_id) <= 0 ||
        H5Iis_valid(uvh5File.DS_header_uvw_array.D_id) <= 0 ||
        H5Iis_valid(uvh5File.DS_header_time_array.D_id) <= 0 ||
        H5Iis_valid(uvh5File.DS_header_integration_time.D_id) <= 0 ||
        H5Iis_valid(uvh5File.DS_phase_center_id_array.D_id) <= 0 ||
        H5Iis_valid(uvh5File.DS_phase_center_app_ra.D_id) <= 0 ||
        H5Iis_valid(uvh5File.DS_phase_center_app_dec.D_id) <= 0 ||
        H5Iis_valid(uvh5File.DS_phase_center_frame_pa.D_id) <= 0 ||
        H5Iis_valid(uvh5File.DS_data_visdata.D_id) <= 0 ||
        H5Iis_valid(uvh5File.DS_data_flags.D_id) <= 0 ||
        H5Iis_valid(uvh5File.DS_data_nsamples.D_id) <= 0 ||
        !uvh5File.flags ||
        !uvh5File.nsamples) {
        JST_ERROR("[MODULE_UVH5_WRITER] UVH5 open failed for '{}': invalid HDF5 file handle.",
                  filepath);
        return Result::ERROR;
    }
    fileOpen = true;

    if (uvh5File.visdata) {
        std::free(uvh5File.visdata);
        uvh5File.visdata = nullptr;
    }

    bytesWritten.publish(0);

    return Result::SUCCESS;
}

Result Uvh5WriterImpl::destroy() {
    Result result = Result::SUCCESS;

    if (openAttempted) {
        uvh5File.visdata = nullptr;
        UVH5close(&uvh5File);
        uvh5File.visdata = nullptr;
        fileOpen = false;
        openAttempted = false;
        headerAllocated = false;
    } else if (headerAllocated) {
        UVH5Hfree(&uvh5File.header);
        headerAllocated = false;
    }

    if (faplOpen) {
        const herr_t status = H5Pclose(faplId);
        if (status < 0) {
            JST_ERROR("[MODULE_UVH5_WRITER] Failed to close the HDF5 file access property list. Error {}.",
                      status);
            result = Result::ERROR;
        }
        faplOpen = false;
        faplId = -1;
    }

    RemovePartialRecording(partialPath);

    uvh5File = {};
    timeline = {};

    return result;
}

Result Uvh5WriterImpl::publishRecording(const BlockTiming& firstBlock) {
    if (partialPath.empty()) {
        return Result::SUCCESS;
    }
    const RecordingGrid grid{firstBlock.timestamp, static_cast<U64>(source.synctime), firstBlock.duration};
    JST_HDF5_CHECK(WriteRecordingGrid(uvh5File.keywords_id, true, grid), [&] {
        JST_ERROR("[MODULE_UVH5_WRITER] Failed to write the UVH5 RF grid keywords. Error {}.", err);
    });
    JST_HDF5_CHECK(H5Fflush(uvh5File.file_id, H5F_SCOPE_GLOBAL), [&] {
        JST_ERROR("[MODULE_UVH5_WRITER] Failed to flush the new recording before publishing it. Error {}.", err);
    });
    if (const auto error = PublishPartialRecording(partialPath, filepath, overwrite)) {
        if (error == std::errc::file_exists) {
            JST_ERROR("[MODULE_UVH5_WRITER] File '{}' already exists.", filepath);
        } else {
            JST_ERROR("[MODULE_UVH5_WRITER] Failed to move the new recording into '{}'.", filepath);
        }
        return Result::ERROR;
    }
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
