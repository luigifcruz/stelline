#include "module_impl.hh"

#include <cstdlib>
#include <cstring>
#include <filesystem>

#include <cufile.h>
#include <hdf5.h>

#include <jetstream/backend/devices/cuda/helpers.hh>

#include "../helpers.hh"

extern "C" {
#include "h5dsc99/h5_dataspace.h"
}

namespace Jetstream::Modules {

struct ObservationBand {
    F64 frequency_start;
    F64 frequency_stop;
    U64 channel_start;
    U64 channel_stop;
    
    JST_SERDES(frequency_start, frequency_stop, channel_start, channel_stop);
};

struct ObservationTuning {
    std::string name;
    std::vector<ObservationBand> bands;

    JST_SERDES(name, bands);
};

struct AntennaCoordinates {
    F64 x;
    F64 y;
    F64 z;

    JST_SERDES(x, y, z);
};

struct AntennaPointing {
    F64 ra;
    F64 dec;
    std::string source_name;

    JST_SERDES(ra, dec);
};

struct AntennaDetails {
    std::string name;
    U64 number;
    F32 diameter;
    AntennaCoordinates position;
    AntennaPointing pointing;
    std::vector<ObservationTuning> tunings;

    JST_SERDES(number, diameter, position);
};

struct ObservationFengine {
    // U64 synctime;
    F64 sample_period;
    
    JST_SERDES(/* synctime, */sample_period);
};

struct ObservationIers {
    // F64 pm_x_arcsec;
    // F64 pm_y_arcsec;
    F64 ut1_utc;
    
    JST_SERDES(/* pm_x_arcsec, pm_y_arcsec, */ut1_utc);
};

struct TelescopeCoordinates {
    F64 latitude;
    F64 longitude;
    F32 altitude;

    JST_SERDES(latitude, longitude, altitude);
};

struct TelescopeInfo {
    std::string name;
    TelescopeCoordinates coordinates;
    std::vector<AntennaDetails> antennas;
    ObservationIers iers;

    JST_SERDES(name, coordinates, antennas, iers);
};

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

Result Uvh5ReaderImpl::publishMetadata(const UVH5_header_t* header, const bool& access_phase_center) {
    TelescopeInfo info;
    info.name = header->telescope_name;
    info.coordinates.latitude = header->latitude;
    info.coordinates.longitude = header->longitude;
    info.coordinates.altitude = header->altitude;
    info.iers.ut1_utc = header->dut1;

    ObservationTuning tuning;
    tuning.name = "Unknown";
    ObservationBand band;
    band.frequency_start = header->freq_array[0]-0.5*header->channel_width[0];
    band.frequency_stop = header->freq_array[0]+0.5*header->channel_width[0];
    band.channel_start = 0;
    band.channel_stop = 1;
    tuning.bands.push_back(band);
    
    if (access_phase_center) {
        // phase_center_id_array is incrementally read,
        // index 0 is always appropriate
        int catalog_index = header->phase_center_id_array[0];
        UVH5_phase_center_t phase_center = header->phase_center_catalog[catalog_index];
        for (size_t index = 0; index < header->Nants_telescope; index++) {
            AntennaDetails ant;
    
            ant.name = header->antenna_names[index];
            ant.number = header->antenna_numbers[index];
            ant.diameter = header->antenna_diameters[index];
            ant.position.x = header->antenna_positions[(3*index)+0];
            ant.position.y = header->antenna_positions[(3*index)+1];
            ant.position.z = header->antenna_positions[(3*index)+2];
            ant.pointing.ra = phase_center.pm_ra;
            ant.pointing.dec = phase_center.pm_dec;
            if (phase_center.info_source != NULL && strlen(phase_center.info_source) > 0) {
                ant.pointing.source_name = phase_center.info_source;
            } else {
                ant.pointing.source_name = "Unknown";
            }
            
            ant.tunings.push_back(tuning);
            info.antennas.push_back(ant);
        }
    }

    if (environment()->set("observatory", info) != Result::SUCCESS) {
        JST_ERROR("[MODULE_UVH5_READER] Could not publish 'observatory' nested environment value.");
        return Result::INCOMPLETE;
    }

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
    JST_CHECK(publishMetadata(&uvh5File.header, false));
    
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
    JST_CHECK(buffer.create(device(), dataType, {batchSize, uvh5File.header.Nbls, uvh5File.DS_data_visdata.dims[1], uvh5File.DS_data_visdata.dims[2]}));
    if (buffer.sizeBytes() != H5DSsize(&uvh5File.DS_data_visdata)) {
        JST_ERROR("[MODULE_UVH5_READER] Signal buffer size is incorrect. {} != {}", buffer.sizeBytes(), H5DSsize(&uvh5File.DS_data_visdata));
        return Result::ERROR;
    }
    JST_CHECK(SetSignalAxes(buffer, {.sample = Index{0}, .channel = Index{2}}));
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
