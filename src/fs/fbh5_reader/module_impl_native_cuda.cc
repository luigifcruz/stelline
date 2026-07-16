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

#include "H5FDgds.h"

namespace Jetstream::Modules {

struct Fbh5ReaderImplNativeCuda : public Fbh5ReaderImpl,
                                  public NativeCudaRuntimeContext,
                                  public Scheduler::Context {
 public:
    Result create() final;
    Result computeSubmit(const cudaStream_t& stream) override;
};

Result Fbh5ReaderImplNativeCuda::create() {
    faplId = H5Pcreate(H5P_FILE_ACCESS);
    if (faplId < 0) {
        JST_ERROR("[MODULE_FBH5_READER] Failed to create the HDF5 file access property list.");
        return Result::ERROR;
    }
    faplOpen = true;

    JST_HDF5_CHECK(H5Pset_fapl_gds(faplId, MBOUNDARY_DEF, FBSIZE_DEF, CBSIZE_DEF), [&] {
        JST_ERROR("[MODULE_FBH5_READER] Failed to configure the HDF5 GDS file access property list. Error {}.", err);
    });

    JST_CHECK(Fbh5ReaderImpl::create());

    return Result::SUCCESS;
}

Result Fbh5ReaderImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    if (!(fbh5File.ds_data.D_id >= 0) || !playing) {
        return Result::SUCCESS;
    }
    
    // We need to synchronize here because the HDF VFD is not asynchronous.
    JST_CUDA_CHECK(cudaStreamSynchronize(stream), [&] {
        JST_ERROR("[MODULE_FBH5_READER_NATIVE_CUDA] Failed to synchronize CUDA stream before FBH5 read: {}.", err);
    });

    const U64 currentIndex = getCurrentBatchIndex();
    herr_t status = filterbank_h5_read(&fbh5File);
    if (status < 0) {
        JST_ERROR("[MODULE_FBH5_READER_NATIVE_CPU] Read failed at offset batch #{} (data index {}).", currentIndex, fbh5File.ds_data.hyperslab_start[0]);
    }
    else if (status == 1) {
        JST_DEBUG("[MODULE_FBH5_READER_NATIVE_CPU] Looping back to start of '{}'.", filepath);
    }

    currentBatchIndex.publish(fbh5File.ds_data.hyperslab_start[0]);
    const U64 actualBytesRead = static_cast<U64>(H5DSsize(&fbh5File.ds_data));
    if (actualBytesRead > 0) {
        updateBandwidth(actualBytesRead);
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(Fbh5ReaderImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
