#ifndef STELLINE_BITS_FRBNN_BASE_HH
#define STELLINE_BITS_FRBNN_BASE_HH

#include <holoscan/holoscan.hpp>

#include <stelline/helpers.hh>
#include <stelline/operators/frbnn/base.hh>

namespace stelline::bits::frbnn {

inline auto FrbnnInferenceBit(auto* app, auto& pool, const std::string& config) {
    using namespace holoscan;
    using namespace stelline::operators::frbnn;

    // Fetch configuration YAML.

    auto frbnn_preprocessor_path = FetchArg<std::string>(app, config, "frbnn_preprocessor_path");
    auto frbnn_path = FetchArg<std::string>(app, config, "frbnn_path");

    // Build FRBNN Preprocessor configuration.

    ops::InferenceOp::DataMap frbnn_preprocessor_path_map;
    frbnn_preprocessor_path_map.insert("frbnn_preprocessor", frbnn_preprocessor_path);

    ops::InferenceOp::DataVecMap frbnn_preprocessor_input_map;
    frbnn_preprocessor_input_map.insert("frbnn_preprocessor", {"input"});

    ops::InferenceOp::DataVecMap frbnn_preprocessor_output_map;
    frbnn_preprocessor_output_map.insert("frbnn_preprocessor", {"output"});

    // Build FRBNN configuration.

    ops::InferenceOp::DataMap frbnn_path_map;
    frbnn_path_map.insert("frbnn", frbnn_path);

    ops::InferenceOp::DataVecMap frbnn_input_map;
    frbnn_input_map.insert("frbnn", {"input"});

    ops::InferenceOp::DataVecMap frbnn_output_map;
    frbnn_output_map.insert("frbnn", {"output"});

    // Instantiate operators.

    auto model_preprocessor = app->template make_operator<ModelPreprocessorOp>(
        "model-preprocessor"
    );
    auto frbnn_preprocessor_inference = app->template make_operator<ops::InferenceOp>(
        "frbnn-preprocessor-inference",
        Arg("backend") = std::string("trt"),
        Arg("model_path_map", frbnn_preprocessor_path_map),
        Arg("pre_processor_map", frbnn_preprocessor_input_map),
        Arg("inference_map", frbnn_preprocessor_output_map),
        Arg("infer_on_cpu") = false,
        Arg("input_on_cuda") = true,
        Arg("output_on_cuda") = true,
        Arg("transmit_on_cuda") = true,
        Arg("is_engine_path") = true,
        Arg("allocator") = pool
    );
    auto model_adapter = app->template make_operator<ModelAdapterOp>(
        "model-adapter"
    );
    auto frbnn_inference = app->template make_operator<ops::InferenceOp>(
        "frbnn-inference",
        Arg("backend") = std::string("trt"),
        Arg("model_path_map", frbnn_path_map),
        Arg("pre_processor_map", frbnn_input_map),
        Arg("inference_map", frbnn_output_map),
        Arg("infer_on_cpu") = false,
        Arg("input_on_cuda") = true,
        Arg("output_on_cuda") = true,
        Arg("transmit_on_cuda") = true,
        Arg("is_engine_path") = true,
        Arg("allocator") = pool
    );
    auto model_postprocessor = app->template make_operator<ModelPostprocessorOp>(
        "model-postprocessor"
    );

    // Connect operators.

    app->add_flow(model_preprocessor, frbnn_preprocessor_inference, {{"out", "receivers"}});
    app->add_flow(frbnn_preprocessor_inference, model_adapter, {{"transmitter", "in"}});
    app->add_flow(model_adapter, frbnn_inference, {{"out", "receivers"}});
    app->add_flow(frbnn_inference, model_postprocessor, {{"transmitter", "in"}});

    return std::pair{model_preprocessor, model_postprocessor};
}

}  // namespace stelline::bits::frbnn

#endif  // STELLINE_BITS_FRBNN_BASE_HH
