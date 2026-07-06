#ifndef STELLINE_FBH5_WRITER_MODULE_HH
#define STELLINE_FBH5_WRITER_MODULE_HH

#include <string>

#include <jetstream/module.hh>

namespace Jetstream::Modules {

struct Fbh5Writer : public Module::Config {
    std::string filepath = "./file.fbh5";
    bool overwrite = false;
    bool recording = false;

    JST_MODULE_TYPE(fbh5_writer);
    JST_MODULE_PARAMS(filepath, overwrite, recording);
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_FBH5_WRITER_MODULE_HH
