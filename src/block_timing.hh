#pragma once

#include <any>
#include <limits>
#include <optional>

#include <jetstream/memory/tensor.hh>
#include <jetstream/types.hh>

namespace Jetstream::Modules {

// `timestamp` is the first F-engine sample of a tensor and `duration` is the
// number of F-engine samples it covers, whatever its shape.
struct BlockTiming {
    U64 timestamp = 0;
    U64 duration = 0;
};

inline std::optional<BlockTiming> ReadBlockTiming(const Tensor& tensor) {
    if (!tensor.hasAttribute("timestamp") || !tensor.hasAttribute("duration")) {
        return std::nullopt;
    }

    const std::any timestamp = tensor.attribute("timestamp");
    const std::any duration = tensor.attribute("duration");
    const auto* timestampValue = std::any_cast<U64>(&timestamp);
    const auto* durationValue = std::any_cast<U64>(&duration);
    if (!timestampValue || !durationValue || *durationValue == 0 ||
        *durationValue > std::numeric_limits<U64>::max() - *timestampValue) {
        return std::nullopt;
    }

    return BlockTiming{*timestampValue, *durationValue};
}

}  // namespace Jetstream::Modules
