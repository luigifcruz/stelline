#include <fcntl.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <hdf5.h>

#include <cassert>

#include <stelline/types.hh>
#include <stelline/operators/filesystem/base.hh>
#include <fmt/format.h>

#include <time.h>

#include "utils/helpers.hh"
#include "utils/modifiers.hh"

extern "C" {
#include "uvh5.h"
#include "uvh5/uvh5_toml.h"
#include "radiointerferometryc99.h"
}

#include "H5FDgds.h"

using namespace gxf;
using namespace holoscan;

namespace stelline::operators::filesystem {

// Replicate sla_Cldj, based on Hatcher (1984) QJRAS 25, 53-55
// https://github.com/scottransom/pyslalib/blob/fcb0650a140a8002cc6c0e8918c3e4c6fe3f8e01/cldj.f
// https://github.com/scottransom/fixbeampos/blob/7b0590a068028eb9ada59678ef0d42640fbdbf4a/cal2mjd.c
/* Validations
  int test_1 = cal2mjd(1999, 07, 10);
  printf("1999-07-10: %d MJD (51639)\n", test_1);
  int test_2 = cal2mjd(1899, 12, 31);
  printf("1899-12-31: %d MJD (15019)\n", test_2);
  int test_3 = cal2mjd(1900, 1, 1);
  printf("1900-01-01: %d MJD (15020)\n", test_3);
  int test_4 = cal2mjd(2000, 1, 1);
  printf("2000-01-01: %d MJD (51544)\n", test_4);
  int test_5 = cal2mjd(2026, 2, 16);
  printf("2026-02-16: %d MJD (61087)\n", test_5);
*/
int cal2mjd(int iy, int im, int id) {
    int leap;

    /* month lengths in days for normal and leap years */
    static int mtab[2][13] = {
        {0,31,28,31,30,31,30,31,31,30,31,30,31},
        {0,31,29,31,30,31,30,31,31,30,31,30,31}
    };

    /*validate year*/
    if (iy < -4699) {
        HOLOSCAN_LOG_ERROR("Invalid year passed to cal2mjd.");
        return 0;
    } else {
        /* validate month */
        if (im < 1 || im > 12) {
            HOLOSCAN_LOG_ERROR("Invalid month passed to cal2mjd.");
            return 0;
        } else {
            /* allow for leap year */
            leap = (iy % 4 == 0 && iy % 100 != 0) || iy % 400 == 0;
            /* validate day */
            if (id < 1 || id > mtab[leap][im]) {
                HOLOSCAN_LOG_ERROR("Invalid day passed to cal2mjd.");
                return 0;
            }
        }
    }

    return (1461 * (iy - (12 - im) / 10 + 4712)) / 4 + (5 + 306 * ((im + 9) % 12)) / 10 - (3 * ((iy - (12 - im) / 10 + 4900) / 100)) / 4 + id - 2399904;
}

struct Uvh5WriterRdmaOp::Impl {
    // Manifest data - populated from manifest on start().

    struct ManifestData {
        // Observatory (static).
        std::string telescope_name;
        double latitude;
        double longitude;
        double altitude;

        // Observation antennas (static, ordered by observation list).
        struct AntennaInfo {
            std::string name;
            uint64_t number;
            double diameter;
            double position[3]; // ECEF x, y, z
        };
        std::vector<AntennaInfo> antennas;

        // Instance subband info (lifetime static).
        std::string instance_bands;
        std::string instance_tuning;
        int instance_band_index;
        double instance_bandcenter;
        double instance_bandwidth;
        uint64_t instance_nof_channels;

        // Pointing (dynamic - first antenna as representative).
        double pointing_ra;
        double pointing_dec;
        std::string pointing_source_name;

        // Timestamp offset (static - first antenna as representative).
        uint64_t packet_timestamp_offset;
        double sample_timespan; // seconds

        // IERS (dynamic).
        double iers_pm_x_arcsec;
        double iers_pm_y_arcsec;
        double iers_ut1_utc;

        bool valid = false;
    } manifestData;

    // State.

    std::string filePath;
    uint64_t dspChannelizationRate;
    uint64_t dspIntegrationRate;
    uint64_t dspFrequencyIntegrationRate;

    // UVH5 file state.

    hid_t faplId;
    UVH5_file_t uvh5_file;
    double integration_timespan; // (s) the timespan of each correlation
    double frequency_bandwidth; // (MHz) the bandwidth of each frequency channel in the correlation
    int64_t bytesWritten;
    uint64_t chunkCounter;
    uint64_t rank;
    std::shared_ptr<holoscan::Tensor> permutedTensor;

    // Metrics.

    std::chrono::time_point<std::chrono::steady_clock> lastMeasurementTime;
    std::atomic<int64_t> bytesSinceLastMeasurement{0};
    std::atomic<double> currentBandwidthMBps{0.0};
};

void Uvh5WriterRdmaOp::initialize() {
    // Allocate memory.
    pimpl = new Impl();

    // Initialize operator.
    Operator::initialize();
}

Uvh5WriterRdmaOp::~Uvh5WriterRdmaOp() {
    delete pimpl;
}

void Uvh5WriterRdmaOp::setup(OperatorSpec& spec) {
    spec.input<std::shared_ptr<holoscan::Tensor>>("in")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));

    spec.param(filePath_, "file_path");
    spec.param(dspChannelizationRate_, "dsp_channelization_rate");
    spec.param(dspIntegrationRate_, "dsp_integration_rate");
    spec.param(dspFrequencyIntegrationRate_, "dsp_frequency_integration_rate");
}

void Uvh5WriterRdmaOp::start() {
    pimpl->filePath = filePath_.get();
    pimpl->dspChannelizationRate = dspChannelizationRate_.get();
    pimpl->dspIntegrationRate = dspIntegrationRate_.get();
    pimpl->dspFrequencyIntegrationRate = dspFrequencyIntegrationRate_.get();

    // Validate and populate manifest data.

    auto& md = pimpl->manifestData;
    md.valid = false;

    assert(manifest() && "UVH5 Writer: manifest provider not set.");
    if (!manifest()) {
        HOLOSCAN_LOG_ERROR("UVH5 Writer: manifest provider not set.");
        return;
    }

    // Observatory.
    manifest()->fetch("observatory.name", md.telescope_name);
    manifest()->fetch("observatory.coordinates.latitude", md.latitude);
    manifest()->fetch("observatory.coordinates.longitude", md.longitude);
    manifest()->fetch("observatory.coordinates.altitude", md.altitude);

    // Observation antennas.
    int32_t nantenna = 0;
    manifest()->fetch("observation.antennas.length", nantenna);
    if (nantenna <= 0) {
        HOLOSCAN_LOG_ERROR("UVH5 Writer: observation.antennas.length is {}", nantenna);
        return;
    }

    md.antennas.resize(nantenna);
    for (int i = 0; i < nantenna; i++) {
        std::string ant_name;
        manifest()->fetch(fmt::format("observation.antennas.{}", i), ant_name);

        md.antennas[i].name = ant_name;
        manifest()->fetch(fmt::format("observatory.antenna.{}.number", ant_name), md.antennas[i].number);
        manifest()->fetch(fmt::format("observatory.antenna.{}.diameter", ant_name), md.antennas[i].diameter);
        manifest()->fetch(fmt::format("observatory.antenna.{}.position.x", ant_name), md.antennas[i].position[0]);
        manifest()->fetch(fmt::format("observatory.antenna.{}.position.y", ant_name), md.antennas[i].position[1]);
        manifest()->fetch(fmt::format("observatory.antenna.{}.position.z", ant_name), md.antennas[i].position[2]);
    }

    // Pointing (from first observation antenna).
    const auto& first_ant = md.antennas[0].name;
    manifest()->fetch(fmt::format("observatory.antenna.{}.pointing.ra", first_ant), md.pointing_ra);
    manifest()->fetch(fmt::format("observatory.antenna.{}.pointing.dec", first_ant), md.pointing_dec);
    manifest()->fetch(fmt::format("observatory.antenna.{}.pointing.source_name", first_ant),
                      md.pointing_source_name);

    // Pointing (from first observation antenna).
    md.packet_timestamp_offset = 0;
    manifest()->fetch(fmt::format("observatory.antenna.{}.fengine.synctime", first_ant),
                      md.packet_timestamp_offset);
    md.sample_timespan = 1e-6;
    manifest()->fetch(fmt::format("observatory.antenna.{}.fengine.sample_period", first_ant),
                      md.sample_timespan);
    pimpl->integration_timespan = md.sample_timespan * pimpl->dspChannelizationRate * pimpl->dspIntegrationRate;

    // IERS.
    manifest()->fetch("observation.iers.pm_x_arcsec", md.iers_pm_x_arcsec);
    manifest()->fetch("observation.iers.pm_y_arcsec", md.iers_pm_y_arcsec);
    manifest()->fetch("observation.iers.ut1_utc", md.iers_ut1_utc);

    // Observation frequency band
    manifest()->fetch("instance.bands", md.instance_bands);
    std::string instance_bands_tuningkey = "'tuning': '";
    std::string instance_bands_indexkey = "'band_index': ";
    size_t tuningkey_pos = md.instance_bands.find(instance_bands_tuningkey);
    size_t indexkey_pos = md.instance_bands.find(instance_bands_indexkey);
    if (tuningkey_pos != std::string::npos && indexkey_pos != std::string::npos) {
        md.instance_tuning = md.instance_bands.substr(tuningkey_pos+instance_bands_tuningkey.length(), 1);
        std::string index_substr = md.instance_bands.substr(indexkey_pos+instance_bands_indexkey.length(), 1);
        md.instance_band_index = std::stoi(index_substr);

        double freq_start, freq_stop;
        uint64_t chan_start, chan_stop;
        manifest()->fetch(
            fmt::format("observatory.antenna.{}.tunings.{}.bands.{}.frequency_start",
                        first_ant,
                        md.instance_tuning,
                        md.instance_band_index),
            freq_start);
        manifest()->fetch(
            fmt::format("observatory.antenna.{}.tunings.{}.bands.{}.frequency_stop",
                        first_ant,
                        md.instance_tuning,
                        md.instance_band_index),
            freq_stop);
        manifest()->fetch(
            fmt::format("observatory.antenna.{}.tunings.{}.bands.{}.channel_start",
                        first_ant,
                        md.instance_tuning,
                        md.instance_band_index),
            chan_start);
        manifest()->fetch(
            fmt::format("observatory.antenna.{}.tunings.{}.bands.{}.channel_stop",
                        first_ant,
                        md.instance_tuning,
                        md.instance_band_index),
            chan_stop);
        md.instance_bandcenter = (freq_stop + freq_start)/2;
        md.instance_bandwidth = freq_stop - freq_start;
        md.instance_nof_channels = chan_stop - chan_start;
        HOLOSCAN_LOG_INFO("UVH5 Writer: ascertained instance band metadata of {} channel(s) ({} MHz wide) centered at {} MHz.", md.instance_nof_channels, md.instance_bandwidth/md.instance_nof_channels, md.instance_bandcenter);
    }
    else {
        HOLOSCAN_LOG_ERROR("UVH5 Writer: could not parse the 'instance.bands' string for metadata: '{}'", md.instance_bands);
    }

    md.valid = true;

    HOLOSCAN_LOG_INFO("UVH5 Writer: manifest loaded ({} antennas, telescope='{}')", nantenna, md.telescope_name);

    // Initialize UVH5.

	pimpl->uvh5_file = {0};
	pimpl->uvh5_file.header = {0};
	UVH5_header_t* uvh5_header = &pimpl->uvh5_file.header;

	// set header scalar data
	uvh5_header->Ntimes = 0; // initially
	uvh5_header->Nfreqs = md.instance_nof_channels * pimpl->dspChannelizationRate / pimpl->dspFrequencyIntegrationRate;
	uvh5_header->Nspws = 1;
	uvh5_header->Nblts = uvh5_header->Nbls * uvh5_header->Ntimes;

    // Populate telescope info from manifest.

    int nant = static_cast<int>(md.antennas.size());
    std::vector<UVH5_antinfo_t> antinfo(nant);
    for (int i = 0; i < nant; i++) {
        antinfo[i].name = const_cast<char*>(md.antennas[i].name.data());
        antinfo[i].number = md.antennas[i].number;
        antinfo[i].position[0] = md.antennas[i].position[0];
        antinfo[i].position[1] = md.antennas[i].position[1];
        antinfo[i].position[2] = md.antennas[i].position[2];
    }

    UVH5set_telescope_info(
        const_cast<char*>(md.telescope_name.data()),
        md.latitude,
        md.longitude,
        md.altitude,
        const_cast<char*>("ecef"),
        md.antennas[0].diameter,
        nant,
        antinfo.data(),
        uvh5_header
    );

    // Populate observation info from manifest.

    std::vector<char*> ant_names_obs(nant);
    for (int i = 0; i < nant; i++) {
        ant_names_obs[i] = const_cast<char*>(md.antennas[i].name.data());
    }

    UVH5set_observation_info(
        nant,
        ant_names_obs.data(),
        const_cast<char*>("xy"),
        uvh5_header
    );

    // Override ant_1/2_array - pipeline does not group auto-baselines first.
    int bl_index = 0;
    for(int a0 = 0; a0 < uvh5_header->Nants_data; a0++) {
        int ant_1_num = uvh5_header->antenna_numbers[a0];
        for(int a1 = a0; a1 < uvh5_header->Nants_data; a1++) {
            int ant_2_num = uvh5_header->antenna_numbers[a1];
            uvh5_header->ant_1_array[bl_index] = ant_1_num;
            uvh5_header->ant_2_array[bl_index] = ant_2_num;
            bl_index++;
        }
    }

	UVH5Hadmin(uvh5_header);
	UVH5Hmalloc_phase_center_catalog(uvh5_header, 1);
    uvh5_header->phase_center_catalog[0].name = const_cast<char*>(md.pointing_source_name.c_str());
	uvh5_header->phase_center_catalog[0].type = UVH5_PHASE_CENTER_SIDEREAL;
    uvh5_header->phase_center_catalog[0].lon = calc_rad_from_degree(md.pointing_ra*360.0/24.0);
    uvh5_header->phase_center_catalog[0].lat = calc_rad_from_degree(md.pointing_dec);
	uvh5_header->phase_center_catalog[0].frame = "icrs";
	uvh5_header->phase_center_catalog[0].epoch = 2000.0;

	uvh5_header->instrument = uvh5_header->telescope_name;
	uvh5_header->history = "None";

	if(uvh5_header->phase_center_catalog[0].type == UVH5_PHASE_CENTER_DRIFTSCAN) {
		memcpy(uvh5_header->_antenna_uvw_positions, uvh5_header->_antenna_enu_positions, sizeof(double)*uvh5_header->Nants_telescope*3);
		UVH5permute_uvws(uvh5_header);
	}
	else if(uvh5_header->phase_center_catalog[0].type == UVH5_PHASE_CENTER_SIDEREAL) {
		memcpy(uvh5_header->_antenna_uvw_positions, uvh5_header->_antenna_enu_positions, sizeof(double)*uvh5_header->Nants_telescope*3);
		double hour_angle_rad = 0.0;
		double declination_rad = 0.0;
		calc_position_to_uvw_frame_from_enu(
			uvh5_header->_antenna_uvw_positions,
			uvh5_header->Nants_data,
			hour_angle_rad,
			declination_rad,
			calc_rad_from_degree(uvh5_header->latitude)
		);

		UVH5permute_uvws(uvh5_header);
	}

	uvh5_header->flex_spw = H5_FALSE;

	uvh5_header->spw_array[0] = 1;
	double instance_subband_lower = md.instance_bandcenter - (md.instance_bandwidth/2);
	pimpl->frequency_bandwidth = md.instance_bandwidth / md.instance_nof_channels;
	pimpl->frequency_bandwidth *= pimpl->dspChannelizationRate / pimpl->dspFrequencyIntegrationRate;
	uvh5_header->channel_width[0] = pimpl->frequency_bandwidth * 1e6;
	for(size_t i = 0; i < uvh5_header->Nfreqs; i++) {
		uvh5_header->channel_width[i] = uvh5_header->channel_width[0];
		uvh5_header->freq_array[i] = (instance_subband_lower + (i+0.5)*pimpl->frequency_bandwidth) * 1e6;
	}

    // Set up HDF5 library.

    pimpl->faplId = H5Pcreate(H5P_FILE_ACCESS);
    HDF5_CHECK_THROW(H5Pset_fapl_gds(pimpl->faplId, MBOUNDARY_DEF, FBSIZE_DEF, CBSIZE_DEF), [&]{
        HOLOSCAN_LOG_ERROR("Error setting the file access property list to H5FD_GDS.");
    });

    // Create HDF5 file.

    UVH5open_with_fileaccess(pimpl->filePath.data(), &pimpl->uvh5_file, UVH5TcreateCF32(), pimpl->faplId);

    if (H5Iis_valid(pimpl->uvh5_file.file_id) <= 0) {
        HOLOSCAN_LOG_ERROR("UVH5 open failed (file='{}'): invalid HDF5 file handle.", pimpl->filePath);
        throw std::runtime_error(fmt::format(
            "UVH5 open failed for '{}': invalid HDF5 file handle.",
            pimpl->filePath
        ));
    }

	UVH5write_keyword_bool(&pimpl->uvh5_file, "keyword_bool", true);
	UVH5write_keyword_double(&pimpl->uvh5_file, "keyword_double", 3.14159265);
	UVH5write_keyword_int(&pimpl->uvh5_file, "keyword_int", 42);
	UVH5write_keyword_string(&pimpl->uvh5_file, "keyword_string", "Testing");

    // Manage UVF5 data pointers
    free(pimpl->uvh5_file.visdata);

    // Reset counters.

    pimpl->chunkCounter = 0;
    pimpl->bytesWritten = 0;
    pimpl->bytesSinceLastMeasurement = 0;
    pimpl->lastMeasurementTime = std::chrono::steady_clock::now();
    pimpl->currentBandwidthMBps = 0.0;
}

void Uvh5WriterRdmaOp::stop() {
    // Close HDF5.
    // replace visdata pointer with something to be freed
    pimpl->uvh5_file.visdata = malloc(8);
    UVH5close(&pimpl->uvh5_file);
}

void Uvh5WriterRdmaOp::compute(InputContext& input, OutputContext&, ExecutionContext&) {
    const auto& meta = metadata();

    auto result = input.receive<std::shared_ptr<holoscan::Tensor>>("in");
    if (!result) {
        return;
    }

    const auto& tensor = result.value();
    const auto& tensorBytes = tensor->size() * (tensor->dtype().bits / 8);

    // Allocate permuted tensor.

    if (pimpl->bytesWritten == 0) {
        CUDA_CHECK_THROW(BlockAlloc(tensor, pimpl->permutedTensor), [&]{
            HOLOSCAN_LOG_ERROR("Failed to allocate permuted tensor.");
        });
    }

    // Permute tensor.

    CUDA_CHECK_THROW(BlockPermutation(pimpl->permutedTensor->to_dlpack(), tensor->to_dlpack()), [&]{
        HOLOSCAN_LOG_ERROR("Failed to permute tensor.");
    });

    // Write tensor to UVH5.
    pimpl->uvh5_file.visdata = pimpl->permutedTensor->data();
    UVH5_header_t* uvh5_header = &pimpl->uvh5_file.header;

    // Refresh dynamic manifest data.
    auto& md = pimpl->manifestData;
    if (md.valid && manifest()) {
        const auto& first_ant = md.antennas[0].name;

        manifest()->fetch(fmt::format("observatory.antenna.{}.pointing.ra", first_ant), md.pointing_ra);
        manifest()->fetch(fmt::format("observatory.antenna.{}.pointing.dec", first_ant), md.pointing_dec);
        manifest()->fetch("observation.iers.pm_x_arcsec", md.iers_pm_x_arcsec);
        manifest()->fetch("observation.iers.pm_y_arcsec", md.iers_pm_y_arcsec);
        manifest()->fetch("observation.iers.ut1_utc", md.iers_ut1_utc);
    }

    double ra_rad = md.pointing_ra;
    double dec_rad = md.pointing_dec;

    // Calculate time point of correlation
    const auto& timestamp = meta->get<uint64_t>("timestamp");
    double realtime_secs = 0.0;
    // Calc real-time seconds since SYNCTIME for pktidx, taken to be a multiple of PKTNTIME:
    //
    //                          pktidx
    //     realtime_secs = -------------------
    //                        1e6 * chan_bw
    if (pimpl->frequency_bandwidth != 0.0) {
        realtime_secs = timestamp / (1e6 * fabs(pimpl->frequency_bandwidth));
    }

    struct timespec ts;
    ts.tv_sec = (time_t)(md.packet_timestamp_offset + rint(realtime_secs));
    ts.tv_nsec = (long)((realtime_secs - rint(realtime_secs)) * 1e9);

    struct tm gmt;
    if (gmtime_r(&ts.tv_sec, &gmt) == NULL) {
        HOLOSCAN_LOG_ERROR("Error calling gmtime_r.");
        return;
    }

    // Gregorian calendar to MJD
    int stt_imjd = cal2mjd(gmt.tm_year+1900, gmt.tm_mon+1, gmt.tm_mday);
    int stt_smjd = gmt.tm_hour*3600 + gmt.tm_min*60+ gmt.tm_sec;
    double stt_offs = ts.tv_nsec*1e-9;

    uvh5_header->time_array[0] = 2400000.5; // JD float offset
    uvh5_header->time_array[0] += (double) stt_imjd;
    uvh5_header->time_array[0] += ((double) stt_smjd) / RADIOINTERFEROMETERY_DAYSEC;
    uvh5_header->time_array[0] += (pimpl->integration_timespan/2) / RADIOINTERFEROMETERY_DAYSEC; // midpoint of integration
	for (size_t i = 0; i < uvh5_header->Nbls; i++)
	{
		uvh5_header->time_array[i] = uvh5_header->time_array[0];
		uvh5_header->integration_time[i] = pimpl->integration_timespan;
	}

    double pos_angle = 0.0;
    int rv;
    if ((rv = calc_itrs_icrs_frame_pos_angle_with_pm_and_ut1_utc(
        uvh5_header->time_array,
        &ra_rad,
        &dec_rad,
        &md.iers_pm_x_arcsec,
        &md.iers_pm_y_arcsec,
        &md.iers_ut1_utc,
        1,
        calc_rad_from_degree(uvh5_header->longitude),
        calc_rad_from_degree(uvh5_header->latitude),
        uvh5_header->altitude,
        RADIOINTERFEROMETERY_PI/360.0, // Offset 0.5 deg, PA is determined over a 1 deg arc.
        &pos_angle
    ))%10 != 0) {
        // RV is Zero if success, otherwise `(index+1)*10+errcode` encoding the index of the
        // erroneous element and the errorcodes:
        // - 0 being dubious year
        // - 1 being unacceptable date.
        // - [2, 8] being radiointerferometry_iers_get() errcode + 3
        HOLOSCAN_LOG_ERROR(
            "Error occurred with radiointerferometryC99.calc_itrs_icrs_frame_pos_angle: rv={}",
            rv
        );
    }

    pimpl->uvh5_file.nsamples[0] = tensor->size();
    for (int i = 0; i < uvh5_header->Nbls; i++) {
        uvh5_header->time_array[i] = uvh5_header->time_array[0];
        // uvh5_header->lst_array[i] = uvh5_header->lst_array[0];
        uvh5_header->phase_center_id_array[i] = 0;
        uvh5_header->phase_center_app_ra[i] = ra_rad;
        uvh5_header->phase_center_app_dec[i] = dec_rad;
        uvh5_header->phase_center_frame_pa[i] = pos_angle;
        for (int j = 0; j < uvh5_header->Nfreqs*uvh5_header->Npols; j++) {
            pimpl->uvh5_file.nsamples[i*uvh5_header->Nfreqs*uvh5_header->Npols + j] = pimpl->uvh5_file.nsamples[0];
            pimpl->uvh5_file.flags[i*uvh5_header->Nfreqs*uvh5_header->Npols + j] = H5_FALSE;
        }
    }

    if (1) { // SIDEREAL
        double dut1 = md.iers_ut1_utc;
        double hour_angle_rad, declination_rad;
        calc_ha_dec_rad(
            ra_rad,
            dec_rad,
            calc_rad_from_degree(uvh5_header->longitude),
            calc_rad_from_degree(uvh5_header->latitude),
            uvh5_header->altitude,
            uvh5_header->time_array[0],
            dut1,
            &hour_angle_rad,
            &declination_rad
        );

        memcpy(uvh5_header->_antenna_uvw_positions, uvh5_header->_antenna_enu_positions, sizeof(double)*uvh5_header->Nants_telescope*3);
        calc_position_to_uvw_frame_from_enu(
            uvh5_header->_antenna_uvw_positions,
            uvh5_header->Nants_telescope,
            hour_angle_rad,
            declination_rad,
            calc_rad_from_degree(uvh5_header->latitude)
        );

        UVH5permute_uvws(uvh5_header);
    }
    UVH5write_dynamic(&pimpl->uvh5_file);

    pimpl->chunkCounter += 1;
    pimpl->bytesWritten += tensorBytes;
    pimpl->bytesSinceLastMeasurement += tensorBytes;
}

void Uvh5WriterRdmaOp::tick() {
    if (!pimpl || !metrics()) {
        return;
    }
    auto now = std::chrono::steady_clock::now();
    auto elapsedSeconds = std::chrono::duration<double>(now - pimpl->lastMeasurementTime).count();

    if (elapsedSeconds > 0.0) {
        int64_t bytes = pimpl->bytesSinceLastMeasurement.exchange(0);
        pimpl->currentBandwidthMBps = static_cast<double>(bytes) / (1024.0 * 1024.0) / elapsedSeconds;
        pimpl->lastMeasurementTime = now;
    }

    metrics()->record("current_bandwidth_mb_s", fmt::format("{:.2f}", pimpl->currentBandwidthMBps.load()));
    metrics()->record("total_data_written_mb", fmt::format("{:.0f}", static_cast<double>(pimpl->bytesWritten) / (1024.0 * 1024.0)));
    metrics()->record("chunks_written", fmt::format("{}", pimpl->chunkCounter));
}

std::string Uvh5WriterRdmaOp::formatMetrics(const MetricsProvider::MetricsMap& metrics) {
    return fmt::format("  Current Bandwidth: {} MB/s\n"
                       "  Total Data Written: {} MB\n"
                       "  Chunks Written: {}",
                       metrics.at("current_bandwidth_mb_s").value,
                       metrics.at("total_data_written_mb").value,
                       metrics.at("chunks_written").value);
}

}  // namespace stelline::operators::io
