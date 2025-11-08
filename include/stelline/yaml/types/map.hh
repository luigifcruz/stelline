#ifndef STELLINE_YAML_MAP_HH
#define STELLINE_YAML_MAP_HH

#include <string>
#include <unordered_map>

#include <stelline/common.hh>
#include <stelline/yaml/utils.hh>

namespace stelline {

typedef std::unordered_map<std::string, std::string> Map;

template<typename T>
inline T FetchMap(const Map& map, const std::string& key, const std::optional<T>& placeholder = {}) {
    auto value = placeholder.value_or(T());

    if (!map.contains(key)) {
        return value;
    }

    if constexpr (std::is_same_v<T, float>) {
        return std::stof(map.at(key));
    } else if constexpr (std::is_same_v<T, int>) {
        return std::stoi(map.at(key));
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return std::stoul(map.at(key));
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return std::stoull(map.at(key));
    } else if constexpr (std::is_same_v<T, bool>) {
        bool result = false;
        if (map.at(key) == "true" || map.at(key) == "True") {
            result = true;
        } else if (map.at(key) == "false" || map.at(key) == "False") {
            result = false;
        } else {
            result = std::stoi(map.at(key)) != 0;
        }
        return result;
    } else if constexpr (std::is_same_v<T, std::string>) {
        return map.at(key);
    }

    throw std::runtime_error("Unsupported type.");
}

}  // namespace stelline

#if STELLINE_IS_NOT_CUDA

#include <holoscan/holoscan.hpp>

template <>
struct YAML::convert<stelline::Map> {
    static Node encode(const stelline::Map& input_spec) {
        throw std::runtime_error("Not implemented.");
    }

    static bool decode(const Node& node, stelline::Map& input_spec) {
        for (auto& [key, element] : stelline::yaml::enumerate_nodes(node)) {
            input_spec[key] = element.as<std::string>();
        }

        return true;
    }
};

#endif  // STELLINE_IS_NOT_CUDA

#endif  // STELLINE_YAML_MAP_HH
