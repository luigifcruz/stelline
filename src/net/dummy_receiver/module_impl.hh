#ifndef STELLINE_DUMMY_RECEIVER_MODULE_IMPL_HH
#define STELLINE_DUMMY_RECEIVER_MODULE_IMPL_HH

#include <stelline/dummy_receiver/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct DummyReceiverImpl : public Module::Impl, public DynamicConfig<DummyReceiver> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;

    const U64& getTimestamp() const;
    const U64& getPeriod() const;

 protected:
    Tensor outputTensor;
    U64 timestamp = 0;
    U64 period = 0;
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_DUMMY_RECEIVER_MODULE_IMPL_HH
