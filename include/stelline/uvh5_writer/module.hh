#ifndef STELLINE_UVH5_WRITER_MODULE_HH
#define STELLINE_UVH5_WRITER_MODULE_HH

#include <string>

#include <jetstream/module.hh>

namespace Jetstream::Modules {

struct Uvh5Writer : public Module::Config {
    std::string filepath = "./file.uvh5";
    bool overwrite = false;
    bool recording = false;

    JST_MODULE_TYPE(stelline_uvh5_writer);
    JST_MODULE_PARAMS(filepath, overwrite, recording);
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_UVH5_WRITER_MODULE_HH
