#ifndef STELLINE_HELPERS_HH
#define STELLINE_HELPERS_HH

#include <string>
#include <holoscan/holoscan.hpp>

#include <stelline/common.hh>

namespace stelline {

template<typename T>
inline T FetchArg(auto* app, const std::string& handle, const std::string& key) {
    auto nodes = app->config().yaml_nodes();

    // Check root.

    if (nodes.empty()) {
        throw std::runtime_error("No configuration nodes found.");
    }

    if (!nodes[0].IsMap()) {
        throw std::runtime_error("Configuration node is not a map.");
    }

    // Check handle.

    if (!nodes[0][handle]) {
        throw std::runtime_error(fmt::format("Configuration node does not contain handle '{}'.", handle));
    }

    if (!nodes[0][handle].IsMap()) {
        throw std::runtime_error(fmt::format("Configuration node '{}' is not a map.", handle));
    }

    // Check key.

    if (!nodes[0][handle][key]) {
        throw std::runtime_error(fmt::format("Configuration node '{}' does not contain key '{}'.", handle, key));
    }

    return nodes[0][handle][key].template as<T>();
}

}  // namespace stelline

#endif  // STELLINE_HELPERS_HH
