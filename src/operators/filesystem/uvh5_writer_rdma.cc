#include <fcntl.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <hdf5.h>

#include <stelline/types.hh>
#include <stelline/operators/filesystem/base.hh>
#include <fmt/format.h>

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

        // Pointing (dynamic - first antenna as representative).
        double pointing_ra;
        double pointing_dec;
        std::string pointing_source_name;

        // IERS (dynamic).
        double iers_pm_x_arcsec;
        double iers_pm_y_arcsec;
        double iers_ut1_utc;

        bool valid = false;
    } manifestData;

    // State.

    std::string filePath;

    // UVH5 file state.

    hid_t faplId;
    UVH5_file_t uvh5_file;
    float tau; // the timespan of each correlation
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
}

void Uvh5WriterRdmaOp::start() {
    pimpl->filePath = filePath_.get();

    // Validate and populate manifest data.

    auto& md = pimpl->manifestData;
    md.valid = false;

    if (!manifest()) {
        HOLOSCAN_LOG_ERROR("UVH5 Writer: manifest provider not set.");
        return;
    }

    bool ok = true;

    auto fetchValue = [&](const std::string& key, auto& out) -> bool {
        auto val = manifest()->fetch(key);
        if (!val.has_value()) {
            HOLOSCAN_LOG_ERROR("UVH5 Writer: missing manifest key '{}'", key);
            ok = false;
            return false;
        }
        using T = std::decay_t<decltype(out)>;
        if (auto* ptr = std::any_cast<T>(&val)) {
            out = *ptr;
            return true;
        }
        HOLOSCAN_LOG_ERROR("UVH5 Writer: manifest key '{}' has unexpected type", key);
        ok = false;
        return false;
    };

    // Observatory.
    fetchValue("observatory.name", md.telescope_name);
    fetchValue("observatory.coordinates.latitude", md.latitude);
    fetchValue("observatory.coordinates.longitude", md.longitude);
    fetchValue("observatory.coordinates.altitude", md.altitude);

    // Observation antennas.
    int32_t nantenna = 0;
    fetchValue("observation.antennas.length", nantenna);
    if (!ok) return;
    if (nantenna <= 0) {
        HOLOSCAN_LOG_ERROR("UVH5 Writer: observation.antennas.length is {}", nantenna);
        return;
    }

    md.antennas.resize(nantenna);
    for (int i = 0; i < nantenna; i++) {
        std::string ant_name;
        fetchValue(fmt::format("observation.antennas.{}", i), ant_name);
        if (!ok) return;

        md.antennas[i].name = ant_name;
        fetchValue(fmt::format("observatory.antenna.{}.number", ant_name), md.antennas[i].number);
        fetchValue(fmt::format("observatory.antenna.{}.diameter", ant_name), md.antennas[i].diameter);
        fetchValue(fmt::format("observatory.antenna.{}.position.x", ant_name), md.antennas[i].position[0]);
        fetchValue(fmt::format("observatory.antenna.{}.position.y", ant_name), md.antennas[i].position[1]);
        fetchValue(fmt::format("observatory.antenna.{}.position.z", ant_name), md.antennas[i].position[2]);
    }
    if (!ok) return;

    // Pointing (from first observation antenna).
    const auto& first_ant = md.antennas[0].name;
    fetchValue(fmt::format("observatory.antenna.{}.pointing.ra", first_ant), md.pointing_ra);
    fetchValue(fmt::format("observatory.antenna.{}.pointing.dec", first_ant), md.pointing_dec);
    fetchValue(fmt::format("observatory.antenna.{}.pointing.source_name", first_ant),
               md.pointing_source_name);

    // IERS.
    fetchValue("observation.iers.pm_x_arcsec", md.iers_pm_x_arcsec);
    fetchValue("observation.iers.pm_y_arcsec", md.iers_pm_y_arcsec);
    fetchValue("observation.iers.ut1_utc", md.iers_ut1_utc);

    if (!ok) return;
    md.valid = true;

    HOLOSCAN_LOG_INFO("UVH5 Writer: manifest loaded ({} antennas, telescope='{}')",
                      nantenna, md.telescope_name);

    // Initialize UVH5.

	pimpl->uvh5_file = {0};
	pimpl->uvh5_file.header = {0};
	UVH5_header_t* uvh5_header = &pimpl->uvh5_file.header;

	// set header scalar data
	uvh5_header->Ntimes = 0; // initially
	uvh5_header->Nfreqs = 192; // TODO placeholder
	uvh5_header->Nspws = 1;
	uvh5_header->Nblts = uvh5_header->Nbls * uvh5_header->Ntimes;

    // Populate telescope info from manifest.

    int nant = static_cast<int>(md.antennas.size());
    std::vector<UVH5_antinfo_t> antinfo(nant);
    for (int i = 0; i < nant; i++) {
        antinfo[i].name = const_cast<char*>(md.antennas[i].name.data());
        antinfo[i].position[0] = md.antennas[i].position[0];
        antinfo[i].position[1] = md.antennas[i].position[1];
        antinfo[i].position[2] = md.antennas[i].position[2];
    }

    UVH5set_telescope_info(
        const_cast<char*>(md.telescope_name.data()),
        md.latitude,
        md.longitude,
        md.altitude,
        const_cast<char*>("ECEF"),
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
        int ant_1_num = uvh5_header->antenna_numbers[
            UVH5find_antenna_index_by_name(uvh5_header, uvh5_header->antenna_names[a0])
        ];
        for(int a1 = a0; a1 < uvh5_header->Nants_data; a1++) {
            int ant_2_num = uvh5_header->antenna_numbers[
                UVH5find_antenna_index_by_name(uvh5_header, uvh5_header->antenna_names[a1])
            ];
            uvh5_header->ant_1_array[bl_index] = ant_1_num;
            uvh5_header->ant_2_array[bl_index] = ant_2_num;
            bl_index++;
        }
    }

	UVH5Hadmin(uvh5_header);
	UVH5Hmalloc_phase_center_catalog(uvh5_header, 1);
    uvh5_header->phase_center_catalog[0].name = const_cast<char*>(md.pointing_source_name.c_str());
	uvh5_header->phase_center_catalog[0].type = UVH5_PHASE_CENTER_SIDEREAL;
	uvh5_header->phase_center_catalog[0].lon = 0.0;
	uvh5_header->phase_center_catalog[0].lat = 0.0;
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

	pimpl->tau = 1.0;
    uvh5_header->time_array[0] = 2400000.5; // JD float offset
    uvh5_header->time_array[0] += 41684.0; // MJD IERS first day
    uvh5_header->time_array[0] += 6.5; // midday of 100th day
    uvh5_header->time_array[0] += (pimpl->tau/2) / RADIOINTERFEROMETERY_DAYSEC; // midpoint of integration
	for (size_t i = 0; i < uvh5_header->Nbls; i++)
	{
		uvh5_header->time_array[i] = uvh5_header->time_array[0];
		uvh5_header->integration_time[i] = pimpl->tau;
	}

    // Set up HDF5 library.

    pimpl->faplId = H5Pcreate(H5P_FILE_ACCESS);
    HDF5_CHECK_THROW(H5Pset_fapl_gds(pimpl->faplId, MBOUNDARY_DEF, FBSIZE_DEF, CBSIZE_DEF), [&]{
        HOLOSCAN_LOG_ERROR("Error setting the file access property list to H5FD_GDS.");
    });

    // Create HDF5 file.

    UVH5open_with_fileaccess(pimpl->filePath.data(), &pimpl->uvh5_file, UVH5TcreateCF32(), pimpl->faplId);
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

        auto ra_val = manifest()->fetch(fmt::format("observatory.antenna.{}.pointing.ra", first_ant));
        auto dec_val = manifest()->fetch(fmt::format("observatory.antenna.{}.pointing.dec", first_ant));
        if (auto* ptr = std::any_cast<double>(&ra_val)) md.pointing_ra = *ptr;
        if (auto* ptr = std::any_cast<double>(&dec_val)) md.pointing_dec = *ptr;

        auto pmx = manifest()->fetch("observation.iers.pm_x_arcsec");
        auto pmy = manifest()->fetch("observation.iers.pm_y_arcsec");
        auto ut1 = manifest()->fetch("observation.iers.ut1_utc");
        if (auto* ptr = std::any_cast<double>(&pmx)) md.iers_pm_x_arcsec = *ptr;
        if (auto* ptr = std::any_cast<double>(&pmy)) md.iers_pm_y_arcsec = *ptr;
        if (auto* ptr = std::any_cast<double>(&ut1)) md.iers_ut1_utc = *ptr;
    }

    double ra_rad = md.pointing_ra;
    double dec_rad = md.pointing_dec;

    uvh5_header->time_array[0] += + pimpl->tau/RADIOINTERFEROMETERY_DAYSEC;
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
