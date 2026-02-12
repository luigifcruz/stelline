#include <memory>
#include <string>

#include <holoscan/holoscan.hpp>

#include <stelline/context.hh>

namespace stelline {

struct MetricsProvider::Impl {
    MetricsProvider::MetricsMap metrics;
};

MetricsProvider::MetricsProvider() {
    pimpl = std::make_unique<Impl>();
}

MetricsProvider::~MetricsProvider() = default;

void MetricsProvider::push(const std::string& key, const std::string& value, bool local) {
    pimpl->metrics[key] = value;
}

MetricsProvider::MetricsMap MetricsProvider::collect() {
    return pimpl->metrics;
}

}  // namespace stelline
