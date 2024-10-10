#include <stelline/types.hh>
#include <stelline/operators/frbnn/base.hh>

using namespace gxf;

namespace stelline::operators::frbnn {

void ModelPreprocessorOp::setup(OperatorSpec& spec) {
    spec.input<DspBlock>("in");
    spec.output<holoscan::gxf::Entity>("out");
}

void ModelPreprocessorOp::compute(InputContext& input, OutputContext& output, ExecutionContext& context) {
    auto block = input.receive<DspBlock>("in").value();

    const auto& meta = metadata();
    meta->set("dsp_block", block);

    auto outMessage = holoscan::gxf::Entity::New(&context);
    outMessage.add(block.tensor, "input");
    output.emit(outMessage, "out");
};

void ModelAdapterOp::setup(OperatorSpec& spec) {
    spec.input<holoscan::gxf::Entity>("in");
    spec.output<holoscan::gxf::Entity>("out");
}

void ModelAdapterOp::compute(InputContext& input, OutputContext& output, ExecutionContext& context) {
    auto inputMessage = input.receive<holoscan::gxf::Entity>("in").value();
    auto inputTensor = inputMessage.get<holoscan::Tensor>("output");

    auto outMessage = holoscan::gxf::Entity::New(&context);
    outMessage.add(inputTensor, "input");
    output.emit(outMessage, "out");
};

void ModelPostprocessorOp::setup(OperatorSpec& spec) {
    spec.input<holoscan::gxf::Entity>("in");
    spec.output<InferenceBlock>("out");
}

void ModelPostprocessorOp::compute(InputContext& input, OutputContext& output, ExecutionContext&) {
    auto inputMessage = input.receive<holoscan::gxf::Entity>("in").value();
    auto inputTensor = inputMessage.get<holoscan::Tensor>("output");

    const auto& meta = metadata();

    InferenceBlock block = {
        .dspBlock = meta->get<DspBlock>("dsp_block"),
        .tensor = inputTensor,
    };

    meta->clear();
    output.emit(block, "out");
};

}  // namespace stelline::operators::frbnn
