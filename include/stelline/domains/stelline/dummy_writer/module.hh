#ifndef STELLINE_DOMAINS_DUMMY_WRITER_MODULE_HH
#define STELLINE_DOMAINS_DUMMY_WRITER_MODULE_HH

#include <jetstream/module.hh>

namespace Jetstream::Modules {

struct DummyWriter : public Module::Config {
    JST_MODULE_TYPE(dummy_writer);
    JST_MODULE_PARAMS();
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_DOMAINS_DUMMY_WRITER_MODULE_HH