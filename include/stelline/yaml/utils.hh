#ifndef STELLINE_YAML_UTILS_HH
#define STELLINE_YAML_UTILS_HH

#include <map>
#include <string>
#include <optional>

#include <stelline/common.hh>

namespace stelline::yaml {

inline std::map<std::string, YAML::Node> enumerate_nodes(const YAML::Node& node, const std::optional<std::string>& key = {}) {
    std::map<std::string, YAML::Node> result;

    if (!node.IsMap()) {
        return result;
    }

    const auto& children = (key.has_value() ? node[key.value()] : node);

    if (!children) {
        return result;
    }

    for (const auto& element : children) {
        std::string key = element.first.as<std::string>();
        result[key] = element.second;
    }

    return result;
}

}  // namespace stelline::yaml

#endif  // STELLINE_YAML_UTILS_HH
