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

struct Uvh5ReaderImplNativeCpu : public Uvh5ReaderImpl,
                                  public NativeCpuRuntimeContext,
                                  public Scheduler::Context {
 public:
    Result create() final;
    Result computeSubmit() override;
};

Result Uvh5ReaderImplNativeCpu::create() {
    JST_CHECK(Uvh5ReaderImpl::create());

    return Result::SUCCESS;
}

Result Uvh5ReaderImplNativeCpu::computeSubmit() {
    if (!(uvh5File.DS_data_visdata.D_id >= 0) || !playing) {
        return Result::SUCCESS;
    }
    const U64 currentIndex = getCurrentBatchIndex();
    herr_t status = UVH5read(&uvh5File);
    if (status < 0) {
        JST_ERROR("[MODULE_UVH5_READER_NATIVE_CPU] Read failed at offset batch #{} (data index {}).", currentIndex, uvh5File.DS_data_visdata.hyperslab_start[0]);
    }
    else if (status == 1) {
        JST_DEBUG("[MODULE_UVH5_READER_NATIVE_CPU] Looping back to start of '{}'.", filepath);
    }

    currentBatchIndex.publish(uvh5File.DS_data_visdata.hyperslab_start[0]/uvh5File.header.Nbls);
    const U64 actualBytesRead = static_cast<U64>(H5DSsize(&uvh5File.DS_data_visdata));
    if (actualBytesRead > 0) {
        updateBandwidth(actualBytesRead);
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(Uvh5ReaderImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
