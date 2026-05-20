#ifndef STELLINE_UVH5_WRITER_MODULE_HH
#define STELLINE_UVH5_WRITER_MODULE_HH

#include <string>

#include <jetstream/module.hh>

namespace Jetstream::Modules {

struct Uvh5Writer : public Module::Config {
    std::string filepath = "./file.uvh5";
    U64 dspChannelizationRate = 1;
    U64 dspIntegrationRate = 1;
    bool overwrite = false;
    bool recording = false;

    JST_MODULE_TYPE(uvh5_writer);
    JST_MODULE_PARAMS(filepath, dspChannelizationRate, dspIntegrationRate, overwrite, recording);
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_UVH5_WRITER_MODULE_HH
