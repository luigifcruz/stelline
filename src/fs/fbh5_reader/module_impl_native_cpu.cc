extern "C" {
#include "h5dsc99/h5_dataspace.h"
}

#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "../helpers.hh"
#include "module_impl.hh"

namespace Jetstream::Modules {

struct Fbh5ReaderImplNativeCpu : public Fbh5ReaderImpl,
                                  public NativeCpuRuntimeContext,
                                  public Scheduler::Context {
 public:
    Result create() final;
    Result computeSubmit() override;
};

Result Fbh5ReaderImplNativeCpu::create() {
    JST_CHECK(Fbh5ReaderImpl::create());

    return Result::SUCCESS;
}

Result Fbh5ReaderImplNativeCpu::computeSubmit() {
    if (!(fbh5File.ds_data.D_id >= 0) || !playing) {
        return Result::SUCCESS;
    }
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

JST_REGISTER_MODULE(Fbh5ReaderImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
