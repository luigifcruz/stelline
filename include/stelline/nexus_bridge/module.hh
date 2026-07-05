#ifndef STELLINE_NEXUS_BRIDGE_MODULE_HH
#define STELLINE_NEXUS_BRIDGE_MODULE_HH

#include <string>

#include <jetstream/module.hh>

namespace Jetstream::Modules {

struct NexusBridge : public Module::Config {
    std::string url = "https://nexus.stelline.space";

    JST_MODULE_TYPE(nexus_bridge);
    JST_MODULE_PARAMS(url);
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_NEXUS_BRIDGE_MODULE_HH
