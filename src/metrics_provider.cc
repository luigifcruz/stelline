#include <memory>
#include <set>
#include <string>

#include <holoscan/holoscan.hpp>

#include <stelline/context.hh>

namespace stelline {

struct MetricsProvider::Impl {
    MetricsProvider::MetricsMap metrics;
    std::set<std::string> globalKeys;
};

MetricsProvider::MetricsProvider() {
    pimpl = std::make_unique<Impl>();
}

MetricsProvider::~MetricsProvider() = default;

void MetricsProvider::record(const std::string& key,
                             const std::string& value,
                             const std::string& type,
                             const bool& global) {
    pimpl->metrics[key] = {type, value};
    if (global) {
        pimpl->globalKeys.insert(key);
    }
}

MetricsProvider::MetricsMap MetricsProvider::snapshot(const bool& global) const {
    if (!global) {
        return pimpl->metrics;
    }
    MetricsMap result;
    for (const auto& [key, value] : pimpl->metrics) {
        if (pimpl->globalKeys.contains(key)) {
            result[key] = value;
        }
    }
    return result;
}

}  // namespace stelline
