#ifndef STELLINE_DETECTOR_MODULE_HH
#define STELLINE_DETECTOR_MODULE_HH

#include <jetstream/module.hh>

namespace Jetstream::Modules {

struct Detector : public Module::Config {
    U64 integrationRate = 1;
    U64 numberOfOutputPolarizations = 4;
    U64 blockSize = 512;

    JST_MODULE_TYPE(detector);
    JST_MODULE_PARAMS(integrationRate, numberOfOutputPolarizations, blockSize);
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_DETECTOR_MODULE_HH
