#ifndef STELLINE_OPERATORS_IO_BASE_HH
#define STELLINE_OPERATORS_IO_BASE_HH

#include <holoscan/holoscan.hpp>

#include <stelline/common.hh>

namespace stelline::operators::io {

using holoscan::Operator;
using holoscan::Parameter;
using holoscan::OperatorSpec;
using holoscan::InputContext;
using holoscan::OutputContext;
using holoscan::ExecutionContext;

class STELLINE_API SimpleSinkOp : public Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(SimpleSinkOp)

    ~SimpleSinkOp();

    void initialize() override;
    void setup(OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
    struct Impl;
    Impl* pimpl;

    Parameter<std::string> filePath_;
    Parameter<bool> enableRdma_;
};

class STELLINE_API DummySinkOp : public Operator {
 public:
      HOLOSCAN_OPERATOR_FORWARD_ARGS(DummySinkOp)
  
      DummySinkOp() = default;
      ~DummySinkOp();
  
      void initialize() override;
      void setup(OperatorSpec& spec) override;
      void start() override;
      void stop() override;
      void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;
  
 private:
      struct Impl;
      Impl* pimpl;
};

}  // namespace stelline::operators::io

#endif  // STELLINE_OPERATORS_IO_BASE_HH
