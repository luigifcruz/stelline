#ifndef STELLINE_FBH5_READER_MODULE_HH
#define STELLINE_FBH5_READER_MODULE_HH

#include <string>

#include <jetstream/module.hh>

namespace Jetstream::Modules {

struct Fbh5Reader : public Module::Config {
    std::string filepath = "";
    U64 batchSize = 256;
    bool loop = true;
    bool playing = true;

    JST_MODULE_TYPE(fbh5_reader);
    JST_MODULE_PARAMS(filepath, batchSize, loop, playing);
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_FBH5_READER_MODULE_HH
