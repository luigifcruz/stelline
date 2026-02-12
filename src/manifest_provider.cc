#include <memory>
#include <string>

#include <holoscan/holoscan.hpp>

#include <stelline/context.hh>

namespace stelline {

struct ManifestProvider::Impl {
    std::string endpoint;
    // TODO: Local cache
};

ManifestProvider::ManifestProvider(const std::string& endpoint) : pimpl(std::make_unique<Impl>()) {
    pimpl->endpoint = endpoint;
    HOLOSCAN_LOG_INFO("ManifestProvider created with endpoint '{}'.", endpoint);
}

ManifestProvider::~ManifestProvider() = default;

std::any ManifestProvider::pull(const std::string& key, uint64_t timestamp) {
    HOLOSCAN_LOG_INFO("ManifestProvider::pull('{}', {}) called.", key, timestamp);
    // TODO: Implement local lookup
    return {};
}

}  // namespace stelline
