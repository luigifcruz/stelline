#ifndef STELLINE_BITS_BLADE_BASE_HH
#define STELLINE_BITS_BLADE_BASE_HH

#include <climits>
#include <unordered_map>

#include <holoscan/holoscan.hpp>

#include <stelline/helpers.hh>
#include <stelline/yaml/types/block_shape.hh>
#include <stelline/yaml/types/map.hh>
#include <stelline/operators/blade/base.hh>

namespace stelline::bits::blade {

inline BitInterface BladeBit(auto* app, auto& pool, uint64_t id, const std::string& config) {
    using namespace holoscan;
    using namespace stelline::operators::blade;

    // Create metadata storage.

    auto metadata = std::make_shared<MetadataStorage>();

    // Fetch configuration YAML.

    auto input_shape = FetchNodeArg<BlockShape>(app, config, "input_shape");
    auto output_shape = FetchNodeArg<BlockShape>(app, config, "output_shape");
    auto mode = FetchNodeArg<std::string>(app, config, "mode");
    auto number_of_buffers = FetchNodeArg<uint64_t>(app, config, "number_of_buffers", 4);
    auto options = FetchNodeArg<Map>(app, config, "options");

    HOLOSCAN_LOG_INFO("Blade Configuration:");
    HOLOSCAN_LOG_INFO("  Input Shape: {}", input_shape);
    HOLOSCAN_LOG_INFO("  Output Shape: {}", output_shape);
    HOLOSCAN_LOG_INFO("  Mode: {}", mode);
    HOLOSCAN_LOG_INFO("  Number of Buffers: {}", number_of_buffers);
    HOLOSCAN_LOG_INFO("  Options: {}", options);

    // Declare modes.

    auto correlator_op_cb = [&](const auto& op_id){
        return app->template make_operator<CorrelatorOp>(
            op_id,
            Arg("number_of_buffers", number_of_buffers),
            Arg("input_shape", input_shape),
            Arg("output_shape", output_shape),
            Arg("options", options)
        );
    };

    auto beamformer_op_cb = [&](const auto& op_id){
        return app->template make_operator<BeamformerOp>(
            op_id,
            Arg("number_of_buffers", number_of_buffers),
            Arg("input_shape", input_shape),
            Arg("output_shape", output_shape),
            Arg("options", options)
        );
    };

    auto frbnn_op_cb = [&](const auto& op_id){
        return app->template make_operator<FrbnnOp>(
            op_id,
            Arg("number_of_buffers", number_of_buffers),
            Arg("input_shape", input_shape),
            Arg("output_shape", output_shape),
            Arg("options", options)
        );
    };

    // Select configuration mode.

    if (mode == "correlator") {
        HOLOSCAN_LOG_INFO("Creating Correlator operator.");
        const auto& correlator_id = fmt::format("blade-correlator-{}", id);
        auto correlator_op = correlator_op_cb(correlator_id);
        correlator_op->load_metadata(correlator_id, metadata);
        return {correlator_op, correlator_op, correlator_op};
    }

    if (mode == "beamformer") {
        HOLOSCAN_LOG_INFO("Creating Beamformer operator.");
        const auto& beamformer_id = fmt::format("blade-beamformer-{}", id);
        auto beamformer_op = beamformer_op_cb(beamformer_id);
        beamformer_op->load_metadata(beamformer_id, metadata);
        return {beamformer_op, beamformer_op, beamformer_op};
    }

    if (mode == "frbnn") {
        HOLOSCAN_LOG_INFO("Creating FRBNN operator.");
        const auto& frbnn_id = fmt::format("blade-frbnn-{}", id);
        auto frbnn_op = frbnn_op_cb(frbnn_id);
        frbnn_op->load_metadata(frbnn_id, metadata);
        return {frbnn_op, frbnn_op, frbnn_op};
    }

    HOLOSCAN_LOG_ERROR("Unsupported mode: {}", mode);
    throw std::runtime_error("Unsupported mode.");
}

}  // namespace stelline::bits::transport

#endif  // STELLINE_BITS_BLADE_BASE_HH
