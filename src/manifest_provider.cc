#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <holoscan/holoscan.hpp>

#include <stelline/context.hh>

namespace stelline {

struct Entry {
    uint64_t start;
    uint64_t end;
    std::any value;
};

struct ManifestProvider::Impl {
    std::unordered_map<std::string, std::vector<Entry>> cache;
};

ManifestProvider::ManifestProvider() {
    pimpl = std::make_unique<Impl>();
}

ManifestProvider::~ManifestProvider() = default;

void ManifestProvider::store(const std::string& key,
                             const std::any& value,
                             const uint64_t& start,
                             const uint64_t& end) {
    pimpl->cache[key].push_back({start, end, value});
}

std::any ManifestProvider::fetch(const std::string& key, const uint64_t& timestamp) const {
    auto it = pimpl->cache.find(key);
    if (it == pimpl->cache.end()) {
        return {};
    }
    for (const auto& entry : it->second) {
        if (timestamp >= entry.start && timestamp < entry.end) {
            return entry.value;
        }
    }
    return {};
}

}  // namespace stelline
