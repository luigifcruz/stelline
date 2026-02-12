#include <memory>
#include <string>

#include <holoscan/holoscan.hpp>

#include <stelline/context.hh>

namespace stelline {

struct MetricsProvider::Impl {
    std::string endpoint;
    MetricsProvider::MetricsMap metrics;
};

MetricsProvider::MetricsProvider(const std::string& endpoint) : pimpl(std::make_unique<Impl>()) {
    pimpl->endpoint = endpoint;
    HOLOSCAN_LOG_INFO("MetricsProvider created with endpoint '{}'.", endpoint);
}

MetricsProvider::~MetricsProvider() = default;

void MetricsProvider::push(const std::string& key, const std::string& value, bool local) {
    pimpl->metrics[key] = value;
}

MetricsProvider::MetricsMap MetricsProvider::collect() {
    return pimpl->metrics;
}

}  // namespace stelline
