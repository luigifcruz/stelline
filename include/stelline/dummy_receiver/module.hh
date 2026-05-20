#ifndef STELLINE_DUMMY_RECEIVER_MODULE_HH
#define STELLINE_DUMMY_RECEIVER_MODULE_HH

#include <string>
#include <vector>

#include <jetstream/module.hh>

namespace Jetstream::Modules {

struct DummyReceiver : public Module::Config {
    std::vector<U64> shape = {1, 1, 1024, 1};
    std::string dataType = "CF32";
    U64 period = 0;

    JST_MODULE_TYPE(dummy_receiver);
    JST_MODULE_PARAMS(shape, dataType, period);
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_DUMMY_RECEIVER_MODULE_HH
