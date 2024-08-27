#ifndef STELLINE_YAML_BLOCKSHAPE_HH
#define STELLINE_YAML_BLOCKSHAPE_HH

#include <vector>

#include <stelline/common.hh>

namespace stelline {

struct BlockShape {
    uint64_t numberOfAntennas;
    uint64_t numberOfChannels;
    uint64_t numberOfSamples;
    uint64_t numberOfPolarizations;
};

}  // namespace stelline

#if STELLINE_IS_NOT_CUDA

#include <holoscan/holoscan.hpp>

template<>
struct fmt::formatter<stelline::BlockShape> {
    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(const stelline::BlockShape& b, FormatContext& ctx) {
        return format_to(ctx.out(), "[A: {}, C: {}, S: {}, P: {}]", b.numberOfAntennas,
                                                                    b.numberOfChannels,
                                                                    b.numberOfSamples,
                                                                    b.numberOfPolarizations);
    }
};

template <>
struct YAML::convert<stelline::BlockShape> {
    static Node encode(const stelline::BlockShape& input_spec) {
        Node node;

        node["number_of_antennas"] = std::to_string(input_spec.numberOfAntennas);
        node["number_of_channels"] = std::to_string(input_spec.numberOfChannels);
        node["number_of_samples"] = std::to_string(input_spec.numberOfSamples);
        node["number_of_polarizations"] = std::to_string(input_spec.numberOfPolarizations);

        return node;
    }

    static bool decode(const Node& node, stelline::BlockShape& input_spec) {
        if (!node.IsMap()) {
            GXF_LOG_ERROR("InputSpec: expected a map");
            return false;
        }

        if (!node["number_of_antennas"] ||
            !node["number_of_channels"] ||
            !node["number_of_samples"]  ||
            !node["number_of_polarizations"]) {
            GXF_LOG_ERROR("InputSpec: missing required fields");
            return false;
        }

        input_spec.numberOfAntennas = node["number_of_antennas"].as<uint64_t>();
        input_spec.numberOfChannels = node["number_of_channels"].as<uint64_t>();
        input_spec.numberOfSamples = node["number_of_samples"].as<uint64_t>();
        input_spec.numberOfPolarizations = node["number_of_polarizations"].as<uint64_t>();

        return true;
    }
};

#endif  // STELLINE_IS_NOT_CUDA

#endif  // STELLINE_YAML_BLOCKSHAPE_HH
