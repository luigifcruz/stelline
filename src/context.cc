#include <memory>
#include <string>

#include <holoscan/holoscan.hpp>

#include <stelline/context.hh>

namespace stelline {

struct Context::Impl {
    ManifestProvider* manifest = nullptr;
    MetricsProvider* metrics = nullptr;
};

Context::Context() : pimpl(std::make_unique<Impl>()) {}

Context::~Context() = default;

void Context::setManifestProvider(ManifestProvider* provider) {
    HOLOSCAN_LOG_INFO("Context wired to ManifestProvider ({}).", provider ? "valid" : "null");
    pimpl->manifest = provider;
}

void Context::setMetricsProvider(MetricsProvider* provider) {
    HOLOSCAN_LOG_INFO("Context wired to MetricsProvider ({}).", provider ? "valid" : "null");
    pimpl->metrics = provider;
}

ManifestProvider* Context::manifest() {
    return pimpl->manifest;
}

MetricsProvider* Context::metrics() {
    return pimpl->metrics;
}

}  // namespace stelline
