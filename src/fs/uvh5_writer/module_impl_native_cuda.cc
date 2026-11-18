#include <cmath>
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

constexpr double kPositionAngleOffset = ERFA_DPI / 360.0;

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

    RecordingGap gap;
    JST_CHECK(timeline.plan(input, "[MODULE_UVH5_WRITER_NATIVE_CUDA]", gap));
    JST_CHECK(CheckRecordingSource(*environment(), "[MODULE_UVH5_WRITER_NATIVE_CUDA]", source));
    const U64 integrations = gap.missingBlocks + 1;
    const auto& header = uvh5File.header;
    if (header.Nbls <= 0 ||
        integrations > static_cast<U64>((std::numeric_limits<int>::max() - header.Nblts) / header.Nbls) ||
        integrations > static_cast<U64>(std::numeric_limits<int>::max() - header.Ntimes)) {
        JST_ERROR("[MODULE_UVH5_WRITER_NATIVE_CUDA] Integrations exceed UVH5 time or baseline count limits.");
        return Result::ERROR;
    }
    const F64 durationSeconds = static_cast<F64>(gap.timing.duration) * metadata.sampleTimespan;

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
            missingTimestamp += gap.timing.duration;
        }
    }
    JST_CHECK(appendIntegration(gap.timing.timestamp, durationSeconds, input.data(), false));
    timeline.commit(gap);
    JST_CHECK(publishRecording(gap.timing));

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
    const F64 longitudeRad = calc_rad_from_degree(header->longitude);
    const F64 latitudeRad = calc_rad_from_degree(header->latitude);
    const F64 julianDate = header->time_array[0];
    F64 apparentRightAscension = 0.0;
    F64 apparentDeclination = 0.0;
    F64 positionAngle = 0.0;
    double hourAngleRad = 0.0;
    double declinationRad = 0.0;

    if (metadata.driftScan) {
        eraAe2hd(metadata.pointingAzimuth, metadata.pointingElevation, latitudeRad, &hourAngleRad, &declinationRad);
        double tai1 = 0.0, tai2 = 0.0, tt1 = 0.0, tt2 = 0.0, ut11 = 0.0, ut12 = 0.0;
        if (eraUtctai(julianDate, 0.0, &tai1, &tai2) < 0 || eraTaitt(tai1, tai2, &tt1, &tt2) != 0 ||
            eraUtcut1(julianDate, 0.0, metadata.iersUt1Utc, &ut11, &ut12) < 0) {
            JST_ERROR("[MODULE_UVH5_WRITER_NATIVE_CUDA] Failed to convert the integration time to UT1 and TT.");
            return Result::ERROR;
        }
        const F64 localSiderealTime = eraGst06a(ut11, ut12, tt1, tt2) + eraSp00(tt1, tt2) + longitudeRad;
        apparentRightAscension = eraAnp(localSiderealTime - hourAngleRad);
        apparentDeclination = declinationRad;
    } else {
        const double polarMotionX = metadata.iersPmXArcsec * ERFA_DAS2R;
        const double polarMotionY = metadata.iersPmYArcsec * ERFA_DAS2R;
        double azimuth = 0.0, zenithDistance = 0.0, observedRightAscension = 0.0, equationOfOrigins = 0.0;
        if (eraAtco13(metadata.pointingRa, metadata.pointingDec, 0.0, 0.0, 0.0, 0.0,
                      julianDate, 0.0, metadata.iersUt1Utc,
                      longitudeRad, latitudeRad, header->altitude,
                      polarMotionX, polarMotionY,
                      0.0, 0.0, 0.0, 0.0,
                      &azimuth, &zenithDistance, &hourAngleRad, &declinationRad,
                      &observedRightAscension, &equationOfOrigins) < 0) {
            JST_ERROR("[MODULE_UVH5_WRITER_NATIVE_CUDA] Failed to compute the apparent phase center.");
            return Result::ERROR;
        }
        apparentRightAscension = eraAnp(observedRightAscension - equationOfOrigins);
        apparentDeclination = declinationRad;

        double icrsRightAscension[2] = {};
        double icrsDeclination[2] = {};
        for (int side = 0; side < 2; ++side) {
            double offsetRightAscension = observedRightAscension;
            double offsetDeclination = declinationRad + (side == 0 ? -kPositionAngleOffset : kPositionAngleOffset);
            if (offsetDeclination > ERFA_DPI / 2.0) {
                offsetRightAscension = eraAnp(offsetRightAscension + ERFA_DPI);
                offsetDeclination = ERFA_DPI - offsetDeclination;
            } else if (offsetDeclination < -ERFA_DPI / 2.0) {
                offsetRightAscension = eraAnp(offsetRightAscension + ERFA_DPI);
                offsetDeclination = -ERFA_DPI - offsetDeclination;
            }
            if (eraAtoc13("R", offsetRightAscension, offsetDeclination,
                          julianDate, 0.0, metadata.iersUt1Utc,
                          longitudeRad, latitudeRad, header->altitude,
                          polarMotionX, polarMotionY,
                          0.0, 0.0, 0.0, 0.0,
                          &icrsRightAscension[side], &icrsDeclination[side]) < 0) {
                JST_ERROR("[MODULE_UVH5_WRITER_NATIVE_CUDA] Failed to compute the phase center position angle.");
                return Result::ERROR;
            }
        }
        positionAngle = -eraPas(icrsRightAscension[0], icrsDeclination[0], icrsRightAscension[1], icrsDeclination[1]);
    }

    const U64 baselineCount = static_cast<U64>(header->Nbls);
    for (U64 baseline = 0; baseline < baselineCount; baseline++) {
        header->phase_center_id_array[baseline] = 0;
        header->phase_center_app_ra[baseline] = apparentRightAscension;
        header->phase_center_app_dec[baseline] = apparentDeclination;
        header->phase_center_frame_pa[baseline] = positionAngle;
    }

    std::memcpy(header->_antenna_uvw_positions,
                header->_antenna_enu_positions,
                sizeof(double) * header->Nants_telescope * 3);
    calc_position_to_uvw_frame_from_enu(header->_antenna_uvw_positions,
                                        header->Nants_telescope,
                                        hourAngleRad,
                                        declinationRad,
                                        latitudeRad);
    UVH5permute_uvws(header);
    const double cosine = std::cos(positionAngle);
    const double sine = std::sin(positionAngle);
    for (U64 baseline = 0; baseline < baselineCount; baseline++) {
        double* uvw = &header->uvw_array[baseline * 3];
        const double u = uvw[0];
        const double v = uvw[1];
        uvw[0] = u * cosine - v * sine;
        uvw[1] = u * sine + v * cosine;
    }

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
