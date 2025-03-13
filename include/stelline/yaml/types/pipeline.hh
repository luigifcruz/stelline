#ifndef STELLINE_YAML_PIPELINE_HH
#define STELLINE_YAML_PIPELINE_HH

#include <vector>

#include <stelline/common.hh>
#include <stelline/yaml/utils.hh>

namespace stelline {

struct PipelineDescriptor {
    struct Node {
        std::string id;
        std::string bit;
        std::string configuration;
        std::unordered_map<std::string, std::string> inputMap;
    };

    std::vector<Node> graph;
};

}  // namespace stelline

#if STELLINE_IS_NOT_CUDA

#include <holoscan/holoscan.hpp>

template<>
struct fmt::formatter<stelline::PipelineDescriptor> {
    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(const stelline::PipelineDescriptor
        & b, FormatContext& ctx) {
        return format_to(ctx.out(), "Graph: {}", b.graph);
    }
};

template<>
struct fmt::formatter<stelline::PipelineDescriptor::Node> {
    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(const stelline::PipelineDescriptor::Node
        & b, FormatContext& ctx) {
        return format_to(ctx.out(), "{{{}: '{}' bit, '{}' configuration, {} inputs}}", b.id, 
                                                                                       b.bit, 
                                                                                       b.configuration, 
                                                                                       b.inputMap);
    }
};

template <>
struct YAML::convert<stelline::PipelineDescriptor> {
    static Node encode(const stelline::PipelineDescriptor& spec) {
        throw std::runtime_error("Not implemented.");
    }

    static bool decode(const Node& head, stelline::PipelineDescriptor& spec) {
        for (const auto& [key, element] :  stelline::yaml::enumerate_nodes(head, "graph")) {
            stelline::PipelineDescriptor::Node node;

            if (!element["bit"] || !element["configuration"]) {
                GXF_LOG_ERROR("InputSpec: missing required fields");
                return false;
            }

            node.id = key;
            node.bit = element["bit"].as<std::string>();
            node.configuration = element["configuration"].as<std::string>();

            for (const auto& [key, element] : stelline::yaml::enumerate_nodes(element, "input")) {
                node.inputMap[key] = element.as<std::string>();
            }

            spec.graph.push_back(node);
        }

        return true;
    }
};

#endif  // STELLINE_IS_NOT_CUDA

#endif  // STELLINE_YAML_PIPELINE_HH
