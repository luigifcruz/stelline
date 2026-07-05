#include <stelline/nexus_bridge/block.hh>
#include <jetstream/detail/block_impl.hh>
#include <stelline/nexus_bridge/module.hh>

#include "module_impl.hh"

namespace Jetstream::Blocks {

namespace {

constexpr const char* kInternalModuleName = "nexus_bridge";

}  // namespace

struct NexusBridgeImpl : public Block::Impl, public DynamicConfig<Blocks::NexusBridge> {
    Result validate() override;
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::NexusBridge> moduleConfig = std::make_shared<Modules::NexusBridge>();
    Modules::NexusBridgeImpl* moduleImpl = nullptr;
};

Result NexusBridgeImpl::validate() {
    const auto& config = *candidate();

    if (device() != DeviceType::CPU) {
        JST_ERROR("[NEXUS_BRIDGE] Block must be created on the CPU device.");
        return Result::ERROR;
    }

    if (runtime() != RuntimeType::PYTHON) {
        JST_ERROR("[NEXUS_BRIDGE] Block must be created with the Python runtime.");
        return Result::ERROR;
    }

    if (config.url.empty()) {
        JST_ERROR("[NEXUS_BRIDGE] Nexus URL must not be empty.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result NexusBridgeImpl::configure() {
    moduleConfig->url = url;

    return Result::SUCCESS;
}

Result NexusBridgeImpl::define() {
    JST_CHECK(defineInterfaceConfig("url",
                                    "Nexus URL",
                                    "Nexus metadata source.",
                                    "text"));

    JST_CHECK(defineInterfaceMetric("connected",
                                    "Connected",
                                    "Whether the bridge has received a Nexus metadata snapshot.",
                                    "private-stelline-metrics-global-number",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->connected.get() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("variablesLoaded",
                                    "Variables Loaded",
                                    "Number of Nexus metadata variables currently mirrored into the flowgraph environment.",
                                    "private-stelline-metrics-global-number",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->variablesLoaded.get() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("connectedDisplay",
                                    "Connected",
                                    "Whether the bridge has received a Nexus metadata snapshot.",
                                    "label",
        [this]() -> std::any {
            const bool isConnected = moduleImpl && moduleImpl->connected.get() != 0;
            return std::any(std::string(isConnected ? "Connected" : "Disconnected"));
        }));

    JST_CHECK(defineInterfaceMetric("variablesLoadedDisplay",
                                    "Variables Loaded",
                                    "Number of Nexus metadata variables currently mirrored into the flowgraph environment.",
                                    "label",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->variablesLoaded.get() : U64(0)));
        }));

    return Result::SUCCESS;
}

Result NexusBridgeImpl::create() {
    JST_CHECK(moduleCreate(kInternalModuleName, moduleConfig, {}));

    moduleImpl = moduleHandle(kInternalModuleName)->getImpl<Modules::NexusBridgeImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(NexusBridgeImpl);

}  // namespace Jetstream::Blocks
