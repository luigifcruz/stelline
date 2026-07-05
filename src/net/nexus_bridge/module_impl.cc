#include "module_impl.hh"

#include <jetstream/flowgraph_environment.hh>

namespace Jetstream::Modules {

namespace {

constexpr const char* kStatusEnvironmentKey = "nexus.bridge";

struct NexusBridgeStatus {
    bool connected = false;
    I64 variables_loaded = 0;

    JST_SERDES(connected, variables_loaded);
};

}  // namespace

Result NexusBridgeImpl::validate() {
    const auto& config = *candidate();

    if (device() != DeviceType::CPU) {
        JST_ERROR("[NEXUS_BRIDGE] Module must be created on the CPU device.");
        return Result::ERROR;
    }

    if (runtime() != RuntimeType::PYTHON) {
        JST_ERROR("[NEXUS_BRIDGE] Module must be created with the Python runtime.");
        return Result::ERROR;
    }

    if (config.url.empty()) {
        JST_ERROR("[NEXUS_BRIDGE] Nexus URL must not be empty.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result NexusBridgeImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::THROTTLED));

    return Result::SUCCESS;
}

Result NexusBridgeImpl::create() {
    connected.publish(0);
    variablesLoaded.publish(0);

    return Result::SUCCESS;
}

Result NexusBridgeImpl::destroy() {
    connected.publish(0);
    variablesLoaded.publish(0);

    return Result::SUCCESS;
}

Result NexusBridgeImpl::reconfigure() {
    return Result::RECREATE;
}

void NexusBridgeImpl::refreshStatus() const {
    NexusBridgeStatus status;
    if (!environment()->tryGet(kStatusEnvironmentKey, status)) {
        connected.publish(0);
        variablesLoaded.publish(0);
        return;
    }

    connected.publish(status.connected ? U64(1) : U64(0));
    variablesLoaded.publish(status.variables_loaded > 0 ? static_cast<U64>(status.variables_loaded) : U64(0));
}

}  // namespace Jetstream::Modules
