#ifndef STELLINE_YAML_UTILS_HH
#define STELLINE_YAML_UTILS_HH

#include <map>
#include <string>

#include <stelline/common.hh>

namespace stelline::yaml {

std::map<std::string, YAML::Node> enumerate_nodes(const YAML::Node& node, const std::string& key) {
    std::map<std::string, YAML::Node> result;
    
    if (!node.IsMap()) {
        return result;
    }

    if (!node[key]) {
        return result;
    }
    
    for (const auto& element : node[key]) {
        std::string key = element.first.as<std::string>();
        result[key] = element.second;
    }
    
    return result;
}

}  // namespace stelline::yaml

#endif  // STELLINE_YAML_UTILS_HH
