#ifndef STELLINE_NEXUS_BRIDGE_MODULE_IMPL_HH
#define STELLINE_NEXUS_BRIDGE_MODULE_IMPL_HH

#include <string>

#include <stelline/nexus_bridge/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/tools/snapshot.hh>

namespace Jetstream::Modules {

struct NexusBridgeImpl : public Module::Impl, public DynamicConfig<NexusBridge> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

    mutable Tools::Snapshot<U64> connected{0};
    mutable Tools::Snapshot<U64> variablesLoaded{0};

 protected:
    void refreshStatus() const;
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_NEXUS_BRIDGE_MODULE_IMPL_HH
