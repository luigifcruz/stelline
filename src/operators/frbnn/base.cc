#include <stelline/operators/frbnn/base.hh>

using namespace gxf;

namespace stelline::operators::frbnn {

void ModelPreprocessorOp::setup(OperatorSpec& spec) {
    spec.input<std::shared_ptr<holoscan::Tensor>>("in");
    spec.output<holoscan::gxf::Entity>("out");
}

void ModelPreprocessorOp::compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) {
    auto input_tensor = op_input.receive<std::shared_ptr<holoscan::Tensor>>("in").value();

    auto out_message = holoscan::gxf::Entity::New(&context);
    out_message.add(input_tensor, "input");
    op_output.emit(out_message, "out");
};

void ModelAdapterOp::setup(OperatorSpec& spec) {
    spec.input<std::shared_ptr<holoscan::Tensor>>("in");
    spec.output<holoscan::gxf::Entity>("out");
}

void ModelAdapterOp::compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) {
    auto in_message = op_input.receive<holoscan::gxf::Entity>("in").value();
    auto in_tensor = in_message.get<holoscan::Tensor>("output");

    auto out_message = holoscan::gxf::Entity::New(&context);
    out_message.add(in_tensor, "input");
    op_output.emit(out_message, "out");
};

void ModelPostprocessorOp::setup(OperatorSpec& spec) {
    spec.input<holoscan::gxf::Entity>("in");
    spec.output<std::shared_ptr<holoscan::Tensor>>("out");
}

void ModelPostprocessorOp::compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) {
    auto in_message = op_input.receive<holoscan::gxf::Entity>("in").value();
    auto in_tensor = in_message.get<holoscan::Tensor>("output");

    op_output.emit(in_tensor, "out");
};

}  // namespace stelline::operatos::frbnn
