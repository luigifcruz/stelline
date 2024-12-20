#ifndef STELLINE_BITS_BLADE_BASE_HH
#define STELLINE_BITS_BLADE_BASE_HH

#include <holoscan/holoscan.hpp>

#include <stelline/helpers.hh>
#include <stelline/yaml/block_shape.hh>
#include <stelline/operators/blade/base.hh>

namespace stelline::bits::blade {

inline auto BladeBit(auto* app, auto& pool, const std::string& config) {
    using namespace holoscan;
    using namespace stelline::operators::blade;

    // Fetch configuration YAML.

    auto input_shape = FetchArg<BlockShape>(app, config, "input_shape");
    auto output_shape = FetchArg<BlockShape>(app, config, "output_shape");
    auto mode = FetchArg<std::string>(app, config, "mode");

    HOLOSCAN_LOG_INFO("Blade Configuration:");
    HOLOSCAN_LOG_INFO("  Input Shape: {}", input_shape);
    HOLOSCAN_LOG_INFO("  Output Shape: {}", output_shape);
    HOLOSCAN_LOG_INFO("  Mode: {}", mode);

    auto correlator_op = [&](){
        // TODO: Implement options parsing.
        // TODO: Implement options printing.

        return app->template make_operator<CorrelatorOp>(
            "correlator",
            Arg("input_shape", input_shape),
            Arg("output_shape", output_shape)
        );
    };

    auto beamformer_op = [&](){
        // TODO: Implement options parsing.
        // TODO: Implement options printing.

        return app->template make_operator<BeamformerOp>(
            "beamformer",
            Arg("input_shape", input_shape),
            Arg("output_shape", output_shape)
        );
    };

    auto frbnn_op = [&](){
        // TODO: Implement options parsing.
        // TODO: Implement options printing.

        return app->template make_operator<FrbnnOp>(
            "frbnn",
            Arg("input_shape", input_shape),
            Arg("output_shape", output_shape)
        );
    };

    if (mode == "correlator") {
        HOLOSCAN_LOG_INFO("Creating Correlator operator.");
        return static_cast<std::shared_ptr<holoscan::Operator>>(correlator_op());
    }

    if (mode == "beamformer") {
        HOLOSCAN_LOG_INFO("Creating Beamformer operator.");
        return static_cast<std::shared_ptr<holoscan::Operator>>(beamformer_op());
    }

    if (mode == "frbnn") {
        HOLOSCAN_LOG_INFO("Creating FRBNN operator.");
        return static_cast<std::shared_ptr<holoscan::Operator>>(frbnn_op());
    }

    HOLOSCAN_LOG_ERROR("Unsupported mode: {}", mode);
    throw std::runtime_error("Unsupported mode.");
}

}  // namespace stelline::bits::transport

#endif  // STELLINE_BITS_BLADE_BASE_HH
