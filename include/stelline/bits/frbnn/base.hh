#ifndef STELLINE_BITS_FRBNN_BASE_HH
#define STELLINE_BITS_FRBNN_BASE_HH

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/inference/inference.hpp>

#include <stelline/helpers.hh>
#include <stelline/operators/frbnn/base.hh>

namespace stelline::bits::frbnn {

inline auto FrbnnInferenceBit(auto* app, auto& pool, const std::string& config) {
    using namespace holoscan;
    using namespace stelline::operators::frbnn;

    // Fetch configuration YAML.

    auto frbnnPreprocessorPath = FetchArg<std::string>(app, config, "frbnn_preprocessor_path");
    auto frbnnPath = FetchArg<std::string>(app, config, "frbnn_path");

    // Build FRBNN Preprocessor configuration.

    ops::InferenceOp::DataMap frbnnPreprocessorPathMap;
    frbnnPreprocessorPathMap.insert("frbnn_preprocessor", frbnnPreprocessorPath);

    ops::InferenceOp::DataVecMap frbnnPreprocessorInputMap;
    frbnnPreprocessorInputMap.insert("frbnn_preprocessor", {"input"});

    ops::InferenceOp::DataVecMap frbnnPreprocessorOutputMap;
    frbnnPreprocessorOutputMap.insert("frbnn_preprocessor", {"output"});

    // Build FRBNN configuration.

    ops::InferenceOp::DataMap frbnnPathMap;
    frbnnPathMap.insert("frbnn", frbnnPath);

    ops::InferenceOp::DataVecMap frbnnInputMap;
    frbnnInputMap.insert("frbnn", {"input"});

    ops::InferenceOp::DataVecMap frbnnOutputMap;
    frbnnOutputMap.insert("frbnn", {"output"});

    // Instantiate operators.

    auto modelPreprocessor = app->template make_operator<ModelPreprocessorOp>(
        "model-preprocessor"
    );
    auto frbnnPreprocessorInference = app->template make_operator<ops::InferenceOp>(
        "frbnn-preprocessor-inference",
        Arg("backend") = std::string("trt"),
        Arg("model_path_map", frbnnPreprocessorPathMap),
        Arg("pre_processor_map", frbnnPreprocessorInputMap),
        Arg("inference_map", frbnnPreprocessorOutputMap),
        Arg("infer_on_cpu") = false,
        Arg("input_on_cuda") = true,
        Arg("output_on_cuda") = true,
        Arg("transmit_on_cuda") = true,
        Arg("is_engine_path") = true,
        Arg("allocator") = pool
    );
    auto modelAdapter = app->template make_operator<ModelAdapterOp>(
        "model-adapter"
    );
    auto frbnnInference = app->template make_operator<ops::InferenceOp>(
        "frbnn-inference",
        Arg("backend") = std::string("trt"),
        Arg("model_path_map", frbnnPathMap),
        Arg("pre_processor_map", frbnnInputMap),
        Arg("inference_map", frbnnOutputMap),
        Arg("infer_on_cpu") = false,
        Arg("input_on_cuda") = true,
        Arg("output_on_cuda") = true,
        Arg("transmit_on_cuda") = true,
        Arg("is_engine_path") = true,
        Arg("allocator") = pool
    );
    auto modelPostprocessor = app->template make_operator<ModelPostprocessorOp>(
        "model-postprocessor"
    );

    // Connect operators.

    app->add_flow(modelPreprocessor, frbnnPreprocessorInference, {{"out", "receivers"}});
    app->add_flow(frbnnPreprocessorInference, modelAdapter, {{"transmitter", "in"}});
    app->add_flow(modelAdapter, frbnnInference, {{"out", "receivers"}});
    app->add_flow(frbnnInference, modelPostprocessor, {{"transmitter", "in"}});

    return std::pair{modelPreprocessor, modelPostprocessor};
}

}  // namespace stelline::bits::frbnn

#endif  // STELLINE_BITS_FRBNN_BASE_HH
