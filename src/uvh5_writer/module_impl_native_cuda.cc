#include <cmath>
#include <cstring>
#include <ctime>

extern "C" {
#include "radiointerferometryc99.h"
}

#include <jetstream/backend/devices/cuda/helpers.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "../net/helpers.hh"
#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

I32 CalToMjd(I32 year, I32 month, I32 day) {
    I32 leapYear = 0;

    static I32 monthTable[2][13] = {
        {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
        {0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
    };

    if (year < -4699 || month < 1 || month > 12) {
        return 0;
    }

    leapYear = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    if (day < 1 || day > monthTable[leapYear][month]) {
        return 0;
    }

    return (1461 * (year - (12 - month) / 10 + 4712)) / 4 +
           (5 + 306 * ((month + 9) % 12)) / 10 -
           (3 * ((year - (12 - month) / 10 + 4900) / 100)) / 4 +
           day - 2399904;
}

}  // namespace

struct Uvh5WriterImplNativeCuda : public Uvh5WriterImpl,
                                  public NativeCudaRuntimeContext,
                                  public Scheduler::Context {
 public:
    Result create() final;
    Result computeSubmit(const cudaStream_t& stream) override;
};

Result Uvh5WriterImplNativeCuda::create() {
    const Tensor& input = inputs().at("input").tensor;

    if (input.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_UVH5_WRITER_NATIVE_CUDA] Unsupported data type '{}'. Expected CF32.", input.dtype());
        return Result::ERROR;
    }

    JST_CHECK(Uvh5WriterImpl::create());

    return Result::SUCCESS;
}

Result Uvh5WriterImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    if (!recording) {
        return Result::SUCCESS;
    }

    const Tensor& input = inputs().at("input").tensor;

    // The HDF5 VFD is host-driven, so make sure upstream CUDA work is visible first.
    JST_CUDA_CHECK(cudaStreamSynchronize(stream), [&] {
        JST_ERROR("[MODULE_UVH5_WRITER_NATIVE_CUDA] Failed to synchronize CUDA stream before UVH5 write: {}.", err);
    });

    if (!fileOpen || !metadata.valid) {
        JST_ERROR("[MODULE_UVH5_WRITER_NATIVE_CUDA] UVH5 file state is not initialized.");
        return Result::ERROR;
    }

    if (input.rank() != expectedShape.size()) {
        JST_ERROR("[MODULE_UVH5_WRITER_NATIVE_CUDA] Input tensor rank changed from {} to {}.",
                  expectedShape.size(),
                  input.rank());
        return Result::ERROR;
    }

    if (input.shape() != expectedShape) {
        JST_ERROR("[MODULE_UVH5_WRITER_NATIVE_CUDA] Input tensor shape changed from {} to {}.",
                  expectedShape,
                  input.shape());
        return Result::ERROR;
    }

    if (input.dtype() != inputDataType) {
        JST_ERROR("[MODULE_UVH5_WRITER_NATIVE_CUDA] Input tensor data type changed from '{}' to '{}'.",
                  inputDataType,
                  input.dtype());
        return Result::ERROR;
    }

    if (!input.contiguous()) {
        JST_ERROR("[MODULE_UVH5_WRITER_NATIVE_CUDA] Input tensor must remain contiguous.");
        return Result::ERROR;
    }

    if (!input.hasAttribute("timestamp")) {
        JST_ERROR("[MODULE_UVH5_WRITER_NATIVE_CUDA] Input tensor missing required 'timestamp' attribute.");
        return Result::ERROR;
    }

    const U64 timestamp = std::any_cast<U64>(input.attribute("timestamp"));
    JST_CHECK(refreshDynamicMetadata(timestamp));

    uvh5File.visdata = const_cast<void*>(input.data());

    UVH5_header_t* header = &uvh5File.header;

    F64 realtimeSeconds = 0.0;
    const F64 channelBandwidth = metadata.instanceBandwidth / static_cast<F64>(metadata.instanceChannelCount);
    if (channelBandwidth != 0.0) {
        realtimeSeconds = static_cast<F64>(timestamp) / (1e6 * std::fabs(channelBandwidth));
    }

    struct timespec timeSpec = {};
    timeSpec.tv_sec = static_cast<time_t>(metadata.packetTimestampOffset + std::llround(realtimeSeconds));
    timeSpec.tv_nsec = static_cast<long>((realtimeSeconds - std::llround(realtimeSeconds)) * 1e9);

    struct tm utcTime = {};
    if (gmtime_r(&timeSpec.tv_sec, &utcTime) == nullptr) {
        JST_ERROR("[MODULE_UVH5_WRITER_NATIVE_CUDA] Failed to convert the UVH5 timestamp to UTC time.");
        return Result::ERROR;
    }

    const I32 mjdDay = CalToMjd(utcTime.tm_year + 1900, utcTime.tm_mon + 1, utcTime.tm_mday);
    if (mjdDay == 0) {
        JST_ERROR("[MODULE_UVH5_WRITER_NATIVE_CUDA] Failed to convert the UVH5 timestamp to MJD.");
        return Result::ERROR;
    }

    const I32 secondsOfDay = utcTime.tm_hour * 3600 + utcTime.tm_min * 60 + utcTime.tm_sec;

    header->time_array[0] = 2400000.5;
    header->time_array[0] += static_cast<F64>(mjdDay);
    header->time_array[0] += static_cast<F64>(secondsOfDay) / RADIOINTERFEROMETERY_DAYSEC;
    header->time_array[0] += (integrationTimespan / 2.0) / RADIOINTERFEROMETERY_DAYSEC;

    for (size_t index = 0; index < static_cast<size_t>(header->Nbls); index++) {
        header->time_array[index] = header->time_array[0];
        header->integration_time[index] = integrationTimespan;
    }

    F64 updatedRightAscensionRad = metadata.pointingRa;
    F64 updatedDeclinationRad = metadata.pointingDec;
    F64 positionAngle = 0.0;
    const I32 rv = calc_itrs_icrs_frame_pos_angle_with_pm_and_ut1_utc(header->time_array,
                                                                       &updatedRightAscensionRad,
                                                                       &updatedDeclinationRad,
                                                                       &metadata.iersPmXArcsec,
                                                                       &metadata.iersPmYArcsec,
                                                                       &metadata.iersUt1Utc,
                                                                       1,
                                                                       calc_rad_from_degree(header->longitude),
                                                                       calc_rad_from_degree(header->latitude),
                                                                       header->altitude,
                                                                       RADIOINTERFEROMETERY_PI / 360.0,
                                                                       &positionAngle);
    if (rv % 10 != 0) {
        JST_ERROR("[MODULE_UVH5_WRITER_NATIVE_CUDA] radiointerferometry position-angle calculation failed: rv={}",
                  rv);
    }

    uvh5File.nsamples[0] = input.size();
    for (I32 baseline = 0; baseline < header->Nbls; baseline++) {
        header->time_array[baseline] = header->time_array[0];
        header->phase_center_id_array[baseline] = 0;
        header->phase_center_app_ra[baseline] = updatedRightAscensionRad;
        header->phase_center_app_dec[baseline] = updatedDeclinationRad;
        header->phase_center_frame_pa[baseline] = positionAngle;

        for (I32 sample = 0; sample < header->Nfreqs * header->Npols; sample++) {
            const I32 sampleIndex = baseline * header->Nfreqs * header->Npols + sample;
            uvh5File.nsamples[sampleIndex] = uvh5File.nsamples[0];
            uvh5File.flags[sampleIndex] = H5_FALSE;
        }
    }

    double hourAngleRad = 0.0;
    double updatedDeclinationForFrame = 0.0;
    calc_ha_dec_rad(updatedRightAscensionRad,
                    updatedDeclinationRad,
                    calc_rad_from_degree(header->longitude),
                    calc_rad_from_degree(header->latitude),
                    header->altitude,
                    header->time_array[0],
                    metadata.iersUt1Utc,
                    &hourAngleRad,
                    &updatedDeclinationForFrame);

    std::memcpy(header->_antenna_uvw_positions,
                header->_antenna_enu_positions,
                sizeof(double) * header->Nants_telescope * 3);
    calc_position_to_uvw_frame_from_enu(header->_antenna_uvw_positions,
                                        header->Nants_telescope,
                                        hourAngleRad,
                                        updatedDeclinationForFrame,
                                        calc_rad_from_degree(header->latitude));
    UVH5permute_uvws(header);

    UVH5write_dynamic(&uvh5File);

    hsize_t currentFileSize = 0;
    JST_HDF5_CHECK(H5Fget_filesize(uvh5File.file_id, &currentFileSize), [&] {
        JST_ERROR("[MODULE_UVH5_WRITER_NATIVE_CUDA] Failed to query the UVH5 file size. Error {}.", err);
    });

    const U64 totalChunks = chunkCounter.get() + 1;
    const U64 previousBytesWritten = bytesWritten.get();
    const U64 totalBytesWritten = static_cast<U64>(currentFileSize);
    const U64 deltaBytesWritten = totalBytesWritten >= previousBytesWritten
                                      ? totalBytesWritten - previousBytesWritten
                                      : 0;

    chunkCounter.publish(totalChunks);
    bytesWritten.publish(totalBytesWritten);
    updateBandwidth(deltaBytesWritten);

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(Uvh5WriterImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
