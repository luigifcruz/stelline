#include <memory>
#include <string>

#include <stelline/manifest.hh>

namespace stelline {

//
// ManifestProvider
//

struct ManifestProvider::Impl {
    std::string endpoint;
    // TODO: gRPC channel, stub, cache
};

ManifestProvider::ManifestProvider(const std::string& endpoint)
    : pimpl(std::make_unique<Impl>()) {
    pimpl->endpoint = endpoint;
}

ManifestProvider::~ManifestProvider() = default;

std::any ManifestProvider::get(const std::string& key, uint64_t timestamp) {
    // TODO: Implement gRPC call + caching
    return {};
}

//
// ManifestConsumer
//

void ManifestConsumer::setManifestProvider(ManifestProvider* provider) {
    manifest_ = provider;
}

ManifestProvider* ManifestConsumer::manifest() {
    return manifest_;
}

}  // namespace stelline
