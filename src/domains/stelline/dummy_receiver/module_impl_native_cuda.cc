#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include <chrono>
#include <thread>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct DummyReceiverImplNativeCuda : public DummyReceiverImpl,
                                     public NativeCudaRuntimeContext,
                                     public Scheduler::Context {
 public:
    Result create() final;
    Result computeSubmit(const cudaStream_t& stream) override;
};

Result DummyReceiverImplNativeCuda::create() {
    JST_CHECK(DummyReceiverImpl::create());

    const DataType dtype = outputTensor.dtype();

    if (dtype != DataType::F32 && dtype != DataType::CF32) {
        JST_ERROR("[MODULE_DUMMY_RECEIVER_NATIVE_CUDA] Unsupported data type '{}'.", dtype);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result DummyReceiverImplNativeCuda::computeSubmit(const cudaStream_t&) {
    timestamp++;
    outputTensor.setAttribute("timestamp", timestamp);

    if (getPeriod() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(getPeriod()));
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(DummyReceiverImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
