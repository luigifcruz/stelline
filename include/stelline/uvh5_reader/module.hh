#ifndef STELLINE_UVH5_READER_MODULE_HH
#define STELLINE_UVH5_READER_MODULE_HH

#include <string>

#include <jetstream/module.hh>

namespace Jetstream::Modules {

struct Uvh5Reader : public Module::Config {
    std::string filepath = "./file.uvh5";
    U64 batchSize = 1;
    bool loop = true;
    bool playing = true;

    JST_MODULE_TYPE(uvh5_reader);
    JST_MODULE_PARAMS(filepath, batchSize, loop, playing);
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_UVH5_READER_MODULE_HH
