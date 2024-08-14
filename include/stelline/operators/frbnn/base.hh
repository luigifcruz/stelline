#ifndef STELLINE_OPERATORS_FRBNN_BASE_HH
#define STELLINE_OPERATORS_FRBNN_BASE_HH

#include <holoscan/holoscan.hpp>

#include <stelline/common.hh>

namespace stelline::operators::frbnn {

using holoscan::Operator;
using holoscan::OperatorSpec;
using holoscan::InputContext;
using holoscan::OutputContext;
using holoscan::ExecutionContext;

class STELLINE_API ModelPreprocessorOp : public Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(ModelPreprocessorOp)

    ModelPreprocessorOp() = default;

    void setup(OperatorSpec& spec) override;
    void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;
};

class STELLINE_API ModelAdapterOp : public Operator {
    public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(ModelAdapterOp)

    ModelAdapterOp() = default;

    void setup(OperatorSpec& spec) override;
    void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;
};

class STELLINE_API ModelPostprocessorOp : public Operator {
    public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(ModelPostprocessorOp)

    ModelPostprocessorOp() = default;

    void setup(OperatorSpec& spec) override;
    void compute(InputContext& input, OutputContext& output, ExecutionContext&) override;
};

}  // namespace stelline::operators::frbnn

#endif  // STELLINE_OPERATORS_FRBNN_BASE_HH
