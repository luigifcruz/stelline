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

#include "H5FDgds.h"

extern "C" {
#include "uvh5.h"
#include "uvh5/uvh5_toml.h"
#include "radiointerferometryc99.h"
}

using namespace gxf;
using namespace holoscan;

namespace stelline::operators::filesystem {

struct Uvh5WriterRdmaOp::Impl {
    // State.

    std::string outputFilePath;
    std::string telinfoFilePath;
    std::string obsantinfoFilePath;
    std::string iersFilePath;

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

    spec.param(output_filePath_, "output_file_path");
    spec.param(telinfo_filePath_, "telinfo_file_path");
    spec.param(obsantinfo_filePath_, "obsantinfo_file_path");
    spec.param(iers_filePath_, "iers_file_path");
}

void Uvh5WriterRdmaOp::start() {
    // Convert Parameters to variables.

    pimpl->outputFilePath = output_filePath_.get();
    pimpl->telinfoFilePath = telinfo_filePath_.get();
    pimpl->obsantinfoFilePath = obsantinfo_filePath_.get();
    pimpl->iersFilePath = iers_filePath_.get();

    // Initialize arrays.
	pimpl->uvh5_file = {0};
	pimpl->uvh5_file.header = {0};
	UVH5_header_t* uvh5_header = &pimpl->uvh5_file.header;

	// set header scalar data
	uvh5_header->Ntimes = 0; // initially
	uvh5_header->Nfreqs = 192; // TODO placeholder
	uvh5_header->Nspws = 1;
	uvh5_header->Nblts = uvh5_header->Nbls * uvh5_header->Ntimes;

	UVH5toml_parse_telescope_info(pimpl->telinfoFilePath.data(), uvh5_header);
	UVH5toml_parse_observation_info(pimpl->obsantinfoFilePath.data(), uvh5_header);

    // BLADE does not group the auto-baselines first
    // TODO using uvh5_header->antenna_names like this assumes Nants_data == Nants_telescope
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
	uvh5_header->phase_center_catalog[0].name = "Center";
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

    UVH5open_with_fileaccess(pimpl->outputFilePath.data(), &pimpl->uvh5_file, UVH5TcreateCF32(), pimpl->faplId);
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
    const auto& tensor = input.receive<std::shared_ptr<holoscan::Tensor>>("in").value();
    const auto& tensorBytes = tensor->size() * (tensor->dtype().bits / 8);

    // Allocate permuted tensor.

    if (pimpl->bytesWritten == 0) {
        CUDA_CHECK_THROW(BlockAlloc(tensor, pimpl->permutedTensor), [&]{
            HOLOSCAN_LOG_ERROR("Failed to allocate permuted tensor.");
        });

        GDS_CHECK_THROW(cuFileBufRegister(pimpl->permutedTensor->data(), tensorBytes, 0), [&]{
            HOLOSCAN_LOG_ERROR("Failed to register buffer with GDS driver.");
        });
    }

    // Permute tensor.

    CUDA_CHECK_THROW(BlockPermutation(pimpl->permutedTensor->to_dlpack(), tensor->to_dlpack()), [&]{
        HOLOSCAN_LOG_ERROR("Failed to permute tensor.");
    });

    // Write tensor to UVH5.
    pimpl->uvh5_file.visdata = pimpl->permutedTensor->data();
    UVH5_header_t* uvh5_header = &pimpl->uvh5_file.header;

    // TODO update from metadata stream
    double ra_rad = 0.628;
    double dec_rad = 0.628;

    uvh5_header->time_array[0] += + pimpl->tau/RADIOINTERFEROMETERY_DAYSEC;
    double pos_angle = 0.0;
    int rv;
    if ((rv = calc_itrs_icrs_frame_pos_angle(
        uvh5_header->time_array,
        &ra_rad,
        &dec_rad,
        1,
        calc_rad_from_degree(uvh5_header->longitude),
        calc_rad_from_degree(uvh5_header->latitude),
        uvh5_header->altitude,
        RADIOINTERFEROMETERY_PI/360.0, // Offset 0.5 deg, PA is determined over a 1 deg arc.
        pimpl->iersFilePath.data(),
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
        float dut1 = 0.0;
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

stelline::StoreInterface::MetricsMap Uvh5WriterRdmaOp::collectMetricsMap() {
    auto now = std::chrono::steady_clock::now();
    auto elapsedSeconds = std::chrono::duration<double>(now - pimpl->lastMeasurementTime).count();

    if (elapsedSeconds > 0.0) {
        int64_t bytes = pimpl->bytesSinceLastMeasurement.exchange(0);
        pimpl->currentBandwidthMBps = static_cast<double>(bytes) / (1024.0 * 1024.0) / elapsedSeconds;
        pimpl->lastMeasurementTime = now;
    }

    stelline::StoreInterface::MetricsMap metrics;
    metrics["current_bandwidth_mb_s"] = fmt::format("{:.2f}", pimpl->currentBandwidthMBps.load());
    metrics["total_data_written_mb"] = fmt::format("{:.0f}", static_cast<double>(pimpl->bytesWritten) / (1024.0 * 1024.0));
    metrics["chunks_written"] = fmt::format("{}", pimpl->chunkCounter);
    return metrics;
}

std::string Uvh5WriterRdmaOp::collectMetricsString() {
    const auto metrics = collectMetricsMap();
    return fmt::format("HDF5 Sink RDMA Operator:\n"
                       "  Current Bandwidth: {} MB/s\n"
                       "  Total Data Written: {} MB\n"
                       "  Chunks Written: {}",
                       metrics.at("current_bandwidth_mb_s"),
                       metrics.at("total_data_written_mb"),
                       metrics.at("chunks_written"));
}

}  // namespace stelline::operators::io
