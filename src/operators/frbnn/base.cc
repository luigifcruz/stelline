#include <stelline/operators/frbnn/base.hh>

using namespace gxf;

namespace stelline::operators::frbnn {

void ModelPreprocessorOp::setup(OperatorSpec& spec) {
    spec.input<std::shared_ptr<holoscan::Tensor>>("in");
    spec.output<holoscan::gxf::Entity>("out");
}

void ModelPreprocessorOp::compute(InputContext& input, OutputContext& output, ExecutionContext& context) {
    auto inputTensor = input.receive<std::shared_ptr<holoscan::Tensor>>("in").value();

    auto outMessage = holoscan::gxf::Entity::New(&context);
    outMessage.add(inputTensor, "input");
    output.emit(outMessage, "out");
};

void ModelAdapterOp::setup(OperatorSpec& spec) {
    spec.input<std::shared_ptr<holoscan::Tensor>>("in");
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
    spec.output<std::shared_ptr<holoscan::Tensor>>("out");
}

void ModelPostprocessorOp::compute(InputContext& input, OutputContext& output, ExecutionContext&) {
    auto inputMessage = input.receive<holoscan::gxf::Entity>("in").value();
    auto inputTensor = inputMessage.get<holoscan::Tensor>("output");

    output.emit(inputTensor, "out");
};

}  // namespace stelline::operators::frbnn
