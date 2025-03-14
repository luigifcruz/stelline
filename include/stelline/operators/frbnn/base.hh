#ifndef STELLINE_OPERATORS_FRBNN_BASE_HH
#define STELLINE_OPERATORS_FRBNN_BASE_HH

#include <holoscan/holoscan.hpp>

#include <stelline/common.hh>

namespace stelline::operators::frbnn {

using holoscan::Operator;
using holoscan::Parameter;
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

class STELLINE_API SimpleDetectionOp : public Operator {
 public:
       HOLOSCAN_OPERATOR_FORWARD_ARGS(SimpleDetectionOp)
   
       ~SimpleDetectionOp();
   
       void initialize() override;
       void setup(OperatorSpec& spec) override;
       void start() override;
       void stop() override;
       void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;
   
 private:
       struct Impl;
       Impl* pimpl;
   
       Parameter<std::string> csvFilePath_;
       Parameter<std::string> hitsDirectory_;
};

}  // namespace stelline::operators::frbnn

#endif  // STELLINE_OPERATORS_FRBNN_BASE_HH
