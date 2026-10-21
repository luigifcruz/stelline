#ifndef STELLINE_POLARIZER_MODULE_HH
#define STELLINE_POLARIZER_MODULE_HH

#include <jetstream/module.hh>

namespace Jetstream::Modules {

struct Polarizer : public Module::Config {
    std::string inputPolarization = "xy";
    std::string outputPolarization = "lr";
    U64 blockSize = 512;

    JST_MODULE_TYPE(polarizer);
    JST_MODULE_PARAMS(inputPolarization, outputPolarization, blockSize);
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_POLARIZER_MODULE_HH
