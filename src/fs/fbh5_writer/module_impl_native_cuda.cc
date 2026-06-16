extern "C" {
#include "h5dsc99/h5_dataspace.h"
}

#include <jetstream/backend/devices/cuda/helpers.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "../helpers.hh"
#include "module_impl.hh"

namespace Jetstream::Modules {

struct Fbh5WriterImplNativeCuda : public Fbh5WriterImpl,
                                  public NativeCudaRuntimeContext,
                                  public Scheduler::Context {
 public:
    Result create() final;
    Result computeSubmit(const cudaStream_t& stream) override;
};

Result Fbh5WriterImplNativeCuda::create() {
    const Tensor& input = inputs().at("input").tensor;

    if (input.dtype() != DataType::F32) {
        JST_ERROR("[MODULE_FBH5_WRITER_NATIVE_CUDA] Unsupported data type '{}'. Expected F32.", input.dtype());
        return Result::ERROR;
    }

    JST_CHECK(Fbh5WriterImpl::create());

    return Result::SUCCESS;
}

Result Fbh5WriterImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    if (!recording) {
        return Result::SUCCESS;
    }

    const Tensor& input = inputs().at("input").tensor;

    // We need to synchronize here because the HDF VFD is not asynchronous.

    JST_CUDA_CHECK(cudaStreamSynchronize(stream), [&] {
        JST_ERROR("[MODULE_FBH5_WRITER_NATIVE_CUDA] Failed to synchronize CUDA stream before FBH5 write: {}.", err);
    });

    if (!fileOpen || !dataOpen || !maskOpen) {
        JST_ERROR("[MODULE_FBH5_WRITER_NATIVE_CUDA] FBH5 file state is not initialized.");
        return Result::ERROR;
    }

    if (input.rank() != expectedShape.size()) {
        JST_ERROR("[MODULE_FBH5_WRITER_NATIVE_CUDA] Input tensor rank changed from {} to {}.",
                  expectedShape.size(),
                  input.rank());
        return Result::ERROR;
    }

    if (input.shape() != expectedShape) {
        JST_ERROR("[MODULE_FBH5_WRITER_NATIVE_CUDA] Input tensor shape changed from {} to {}.",
                  expectedShape,
                  input.shape());
        return Result::ERROR;
    }

    if (input.dtype() != inputDataType) {
        JST_ERROR("[MODULE_FBH5_WRITER_NATIVE_CUDA] Input tensor data type changed from '{}' to '{}'.",
                  inputDataType,
                  input.dtype());
        return Result::ERROR;
    }

    if (!input.contiguous()) {
        JST_ERROR("[MODULE_FBH5_WRITER_NATIVE_CUDA] Input tensor must remain contiguous.");
        return Result::ERROR;
    }

    const void* inputPtr = input.data();
    const U64 inputBytes = input.sizeBytes();

    if (!bufferRegistered || registeredBuffer != inputPtr || registeredBufferSize != inputBytes) {
        if (bufferRegistered) {
            JST_CUFILE_CHECK(cuFileBufDeregister(const_cast<void*>(registeredBuffer)), [&] {
                JST_ERROR("[MODULE_FBH5_WRITER_NATIVE_CUDA] Failed to deregister the previous input buffer (CUfile error {}).",
                          err);
            });
            bufferRegistered = false;
            registeredBuffer = nullptr;
            registeredBufferSize = 0;
        }

        JST_CUFILE_CHECK(cuFileBufRegister(const_cast<void*>(inputPtr), inputBytes, 0), [&] {
            JST_ERROR("[MODULE_FBH5_WRITER_NATIVE_CUDA] Failed to register the input buffer with GDS (CUfile error {}).",
                      err);
        });

        bufferRegistered = true;
        registeredBuffer = inputPtr;
        registeredBufferSize = inputBytes;
    }

    JST_HDF5_CHECK(H5DSextend(&fbh5File.ds_data), [&] {
        JST_ERROR("[MODULE_FBH5_WRITER_NATIVE_CUDA] Failed to extend the FBH5 data dataset. Error {}.", err);
    });

    JST_HDF5_CHECK(H5DSwrite(&fbh5File.ds_data, const_cast<void*>(inputPtr)), [&] {
        JST_ERROR("[MODULE_FBH5_WRITER_NATIVE_CUDA] Failed to write the FBH5 data dataset. Error {}.", err);
    });

    JST_HDF5_CHECK(H5DSextend(&fbh5File.ds_mask), [&] {
        JST_ERROR("[MODULE_FBH5_WRITER_NATIVE_CUDA] Failed to extend the FBH5 mask dataset. Error {}.", err);
    });

    JST_HDF5_CHECK(H5DSwrite(&fbh5File.ds_mask, mask), [&] {
        JST_ERROR("[MODULE_FBH5_WRITER_NATIVE_CUDA] Failed to write the FBH5 mask dataset. Error {}.", err);
    });

    hsize_t currentFileSize = 0;
    JST_HDF5_CHECK(H5Fget_filesize(fbh5File.file_id, &currentFileSize), [&] {
        JST_ERROR("[MODULE_FBH5_WRITER_NATIVE_CUDA] Failed to query the FBH5 file size. Error {}.", err);
    });

    const U64 totalChunks = chunkCounter.get() + 1;
    const U64 previousBytesWritten = bytesWritten.get();
    const U64 totalBytesWritten = static_cast<U64>(currentFileSize);
    const U64 deltaBytesWritten = totalBytesWritten >= previousBytesWritten
                                      ? totalBytesWritten - previousBytesWritten
                                      : 0;

    chunkCounter.publish(totalChunks);
    bytesWritten.publish(totalBytesWritten);
    bytesSinceLastMeasurement += static_cast<I64>(deltaBytesWritten);

    const auto now = std::chrono::steady_clock::now();
    const F64 elapsedSeconds = std::chrono::duration<F64>(now - lastMeasurementTime).count();
    if (elapsedSeconds >= 0.10) {
        const F64 instantMBps = elapsedSeconds > 0.0
                                    ? static_cast<F64>(bytesSinceLastMeasurement) / (1024.0 * 1024.0) / elapsedSeconds
                                    : 0.0;

        constexpr F64 kEmaAlpha = 0.3;
        bandwidthMBps.publish(kEmaAlpha * instantMBps +
                              (1.0 - kEmaAlpha) * bandwidthMBps.get());

        bytesSinceLastMeasurement = 0;
        lastMeasurementTime = now;
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(Fbh5WriterImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
