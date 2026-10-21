#ifndef STELLINE_BEAMFORMER_MODULE_HH
#define STELLINE_BEAMFORMER_MODULE_HH

#include <jetstream/module.hh>

namespace Jetstream::Modules {

struct Beamformer : public Module::Config {
    bool enableIncoherentBeam = false;
    bool enableIncoherentBeamSqrt = false;
    U64 blockSize = 512;

    JST_MODULE_TYPE(beamformer);
    JST_MODULE_PARAMS(enableIncoherentBeam, enableIncoherentBeamSqrt, blockSize);
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_BEAMFORMER_MODULE_HH
