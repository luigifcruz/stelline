#include <memory>
#include <string>

#include <holoscan/holoscan.hpp>

#include <stelline/context.hh>

namespace stelline {

struct ManifestProvider::Impl {

};

ManifestProvider::ManifestProvider() {
    pimpl = std::make_unique<Impl>();
}

ManifestProvider::~ManifestProvider() = default;

std::any ManifestProvider::pull(const std::string& key, uint64_t timestamp) {
    HOLOSCAN_LOG_INFO("ManifestProvider::pull('{}', {}) called.", key, timestamp);
    // TODO: Implement local lookup
    return {};
}

}  // namespace stelline
