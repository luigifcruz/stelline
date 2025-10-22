#ifndef STELLINE_BITS_FRBNN_BASE_HH
#define STELLINE_BITS_FRBNN_BASE_HH

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/inference/inference.hpp>

#include <stelline/helpers.hh>
#include <stelline/operators/frbnn/base.hh>

namespace stelline::bits::frbnn {

//
// FrbnnInferenceBit
//

inline BitInterface FrbnnInferenceBit(auto* app, auto& pool, uint64_t id, const std::string& config) {
    using namespace holoscan;
    using namespace stelline::operators::frbnn;



    // Configure app.



    // Fetch configuration YAML.

    auto frbnn_preprocessor_path = FetchNodeArg<std::string>(app, config, "frbnn_preprocessor_path");
    auto frbnn_path = FetchNodeArg<std::string>(app, config, "frbnn_path");

    HOLOSCAN_LOG_INFO("FRBNN Inference Configuration:");
    HOLOSCAN_LOG_INFO("  Preprocessor Path: {}", frbnn_preprocessor_path);
    HOLOSCAN_LOG_INFO("  Model Path: {}", frbnn_path);

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

    const auto& model_preprocessor_id = fmt::format("frbnn-model-preprocessor-{}", id);
    auto model_preprocessor_op = app->template make_operator<ModelPreprocessorOp>(
        model_preprocessor_id
    );


    const auto& frbnn_preprocessor_inference_id = fmt::format("frbnn-preprocessor-inference-{}", id);
    auto frbnn_preprocessor_inference_op = app->template make_operator<ops::InferenceOp>(
        frbnn_preprocessor_inference_id,
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

    const auto& model_adapter_id = fmt::format("frbnn-model-adapter-{}", id);
    auto model_adapter_op = app->template make_operator<ModelAdapterOp>(
        model_adapter_id
    );


    const auto& frbnn_inference_id = fmt::format("frbnn-inference-{}", id);
    auto frbnn_inference_op = app->template make_operator<ops::InferenceOp>(
        frbnn_inference_id,
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

    const auto& model_postprocessor_id = fmt::format("frbnn-model-postprocessor-{}", id);
    auto model_postprocessor_op = app->template make_operator<ModelPostprocessorOp>(
        model_postprocessor_id
    );


    // Connect operators.

    app->add_flow(model_preprocessor_op, frbnn_preprocessor_inference_op, {{"out", "receivers"}});
    app->add_flow(frbnn_preprocessor_inference_op, model_adapter_op, {{"transmitter", "in"}});
    app->add_flow(model_adapter_op, frbnn_inference_op, {{"out", "receivers"}});
    app->add_flow(frbnn_inference_op, model_postprocessor_op, {{"transmitter", "in"}});

    return {model_preprocessor_op, model_postprocessor_op};
}

//
// FrbnnDetectionBit
//

inline BitInterface FrbnnDetectionBit(auto* app, auto& pool, uint64_t id, const std::string& config) {
    using namespace holoscan;
    using namespace stelline::operators::frbnn;



    // Fetch configuration YAML.

    auto frbnn_csv_file_path = FetchNodeArg<std::string>(app, config, "csv_file_path");
    auto frbnn_hits_directory = FetchNodeArg<std::string>(app, config, "hits_directory");

    HOLOSCAN_LOG_INFO("FRBNN Detection Configuration:");
    HOLOSCAN_LOG_INFO("  CSV File Path: {}", frbnn_csv_file_path);
    HOLOSCAN_LOG_INFO("  Hits Directory: {}", frbnn_hits_directory);

    // Instantiate operators.

    const auto& frbnn_simple_detection_id = fmt::format("frbnn-simple-detection-{}", id);
    auto frbnn_simple_detection_op = app->template make_operator<SimpleDetectionOp>(
        frbnn_simple_detection_id,
        Arg("csv_file_path", frbnn_csv_file_path),
        Arg("hits_directory", frbnn_hits_directory)
    );


    return {frbnn_simple_detection_op, frbnn_simple_detection_op};
}

}  // namespace stelline::bits::frbnn

#endif  // STELLINE_BITS_FRBNN_BASE_HH
