#include <cstring>
#include <limits>
#include <new>
#include <vector>

extern "C" {
#include "radiointerferometryc99.h"
}

#include <jetstream/backend/devices/cuda/helpers.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "../helpers.hh"
#include "module_impl.hh"

namespace Jetstream::Modules {

struct Uvh5WriterImplNativeCuda : public Uvh5WriterImpl,
                                   public NativeCudaRuntimeContext,
                                   public Scheduler::Context {
 public:
    Result validate() final;
    Result computeSubmit(const cudaStream_t& stream) override;

 private:
    Result appendIntegration(U64 timestamp, F64 durationSeconds, const void* data, bool missing);
};

Result Uvh5WriterImplNativeCuda::validate() {
    JST_CHECK(Uvh5WriterImpl::validate());

    if (!inputs().contains("input")) {
        return Result::SUCCESS;
    }

    const Tensor& input = inputs().at("input").tensor;
    if (!input.validShape() || input.size() == 0) {
        return Result::SUCCESS;
    }

    if (input.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_UVH5_WRITER_NATIVE_CUDA] Unsupported data type '{}'. Expected CF32.", input.dtype());
        return Result::ERROR;
    }

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

    const std::any timestampValue = input.attribute("timestamp");
    const auto* timestamp = std::any_cast<U64>(&timestampValue);
    if (!timestamp) {
        JST_ERROR("[MODULE_UVH5_WRITER_NATIVE_CUDA] Input 'timestamp' attribute must have type U64.");
        return Result::ERROR;
    }
    RecordingGap gap;
    JST_CHECK(timeline.plan(input, gap));
    JST_CHECK(checkClockEpoch());
    const U64 integrations = gap.missingBlocks + 1;
    const auto& header = uvh5File.header;
    if (header.Nbls <= 0 ||
        integrations > static_cast<U64>((std::numeric_limits<int>::max() - header.Nblts) / header.Nbls) ||
        integrations > static_cast<U64>(std::numeric_limits<int>::max() - header.Ntimes)) {
        JST_ERROR("[MODULE_UVH5_WRITER_NATIVE_CUDA] Integrations exceed UVH5 time or baseline count limits.");
        return Result::ERROR;
    }
    const F64 durationSeconds = gap.timing
        ? static_cast<F64>(gap.timing->duration) * metadata.sampleTimespan
        : integrationTimespan;

    if (gap.missingBlocks != 0) {
        std::vector<CF32> missingData;
        try {
            missingData.resize(input.size(), CF32{});
        } catch (const std::bad_alloc&) {
            JST_ERROR("[MODULE_UVH5_WRITER_NATIVE_CUDA] Failed to allocate the missing-visibility buffer.");
            return Result::ERROR;
        }
        U64 missingTimestamp = gap.firstMissing;
        for (U64 block = 0; block < gap.missingBlocks; ++block) {
            JST_CHECK(appendIntegration(missingTimestamp, durationSeconds, missingData.data(), true));
            missingTimestamp += gap.timing->duration;
        }
    }
    JST_CHECK(appendIntegration(*timestamp, durationSeconds, input.data(), false));
    timeline.commit(gap);

    hsize_t currentFileSize = 0;
    JST_HDF5_CHECK(H5Fget_filesize(uvh5File.file_id, &currentFileSize), [&] {
        JST_ERROR("[MODULE_UVH5_WRITER_NATIVE_CUDA] Failed to query the UVH5 file size. Error {}.", err);
    });

    const U64 previousBytesWritten = bytesWritten.get();
    const U64 totalBytesWritten = static_cast<U64>(currentFileSize);
    const U64 deltaBytesWritten = totalBytesWritten >= previousBytesWritten
                                      ? totalBytesWritten - previousBytesWritten
                                      : 0;

    chunkCounter.publish(chunkCounter.get() + integrations);
    bytesWritten.publish(totalBytesWritten);
    updateBandwidth(deltaBytesWritten);

    return Result::SUCCESS;
}

Result Uvh5WriterImplNativeCuda::appendIntegration(const U64 timestamp,
                                                   const F64 durationSeconds,
                                                   const void* data,
                                                   const bool missing) {
    JST_CHECK(prepareIntegration(timestamp, durationSeconds, missing));
    JST_CHECK(refreshDynamicMetadata(timestamp));

    UVH5_header_t* header = &uvh5File.header;
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
        JST_ERROR("[MODULE_UVH5_WRITER_NATIVE_CUDA] The radiointerferometry position-angle calculation failed: rv={}",
                  rv);
        return Result::ERROR;
    }

    const U64 baselineCount = static_cast<U64>(header->Nbls);
    for (U64 baseline = 0; baseline < baselineCount; baseline++) {
        header->phase_center_id_array[baseline] = 0;
        header->phase_center_app_ra[baseline] = updatedRightAscensionRad;
        header->phase_center_app_dec[baseline] = updatedDeclinationRad;
        header->phase_center_frame_pa[baseline] = positionAngle;
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

    uvh5File.visdata = const_cast<void*>(data);
    const auto status = UVH5write_dynamic(&uvh5File);
    uvh5File.visdata = nullptr;
    if (status != 0) {
        JST_ERROR("[MODULE_UVH5_WRITER_NATIVE_CUDA] Failed to write the UVH5 datasets.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(Uvh5WriterImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
