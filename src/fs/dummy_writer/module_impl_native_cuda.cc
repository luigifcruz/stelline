#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct DummyWriterImplNativeCuda : public DummyWriterImpl,
                                   public NativeCudaRuntimeContext,
                                   public Scheduler::Context {
 public:
    Result create() final;
    Result computeSubmit(const cudaStream_t& stream) override;
};

Result DummyWriterImplNativeCuda::create() {
    JST_CHECK(DummyWriterImpl::create());
    return Result::SUCCESS;
}

Result DummyWriterImplNativeCuda::computeSubmit(const cudaStream_t&) {
    const auto& input = inputs().at("input").tensor;

    if (input.hasAttribute("timestamp")) {
        latestTimestamp = std::any_cast<U64>(input.attribute("timestamp"));
    }

    iterationCount++;

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(DummyWriterImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
