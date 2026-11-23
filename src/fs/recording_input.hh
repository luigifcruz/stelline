#pragma once

#include <optional>
#include <vector>

#include <jetstream/logger.hh>
#include <jetstream/memory/axis.hh>
#include <jetstream/memory/tensor.hh>
#include <jetstream/types.hh>

namespace Jetstream::Modules {

struct RecordingInputPlan {
    std::optional<std::vector<U64>> permutation;
    bool copy = false;
};

inline Result PlanRecordingInput(const Tensor& input, const char* tag, RecordingInputPlan& plan) {
    plan = {};
    if (!input.validShape() || input.size() == 0) {
        return Result::SUCCESS;
    }

    SignalAxes axes;
    if (MapSignalAxes(input, IdentityAxisMap(input.rank()), axes) != Result::SUCCESS) {
        JST_ERROR("{} Input signal axis metadata is invalid.", tag);
        return Result::ERROR;
    }

    if (axes.sample && *axes.sample != 0) {
        std::vector<U64> order = {static_cast<U64>(*axes.sample)};
        for (U64 axis = 0; axis < input.rank(); ++axis) {
            if (axis != static_cast<U64>(*axes.sample)) {
                order.push_back(axis);
            }
        }
        plan.permutation = std::move(order);
    }

    plan.copy = plan.permutation.has_value() || !input.contiguous() ||
                input.buffer().location() != Location::Device;
    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
