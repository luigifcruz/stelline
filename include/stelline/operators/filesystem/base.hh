#ifndef STELLINE_OPERATORS_FILESYSTEM_BASE_HH
#define STELLINE_OPERATORS_FILESYSTEM_BASE_HH

#include <holoscan/holoscan.hpp>

#include <stelline/common.hh>
#include <stelline/metadata.hh>

namespace stelline::operators::filesystem {

using holoscan::Operator;
using holoscan::Parameter;
using holoscan::OperatorSpec;
using holoscan::InputContext;
using holoscan::OutputContext;
using holoscan::ExecutionContext;

class STELLINE_API SimpleWriterOp : public Operator, public Metadata {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(SimpleWriterOp)

    ~SimpleWriterOp();

    void initialize() override;
    void setup(OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
    struct Impl;
    Impl* pimpl;

    Parameter<std::string> filePath_;
};

class STELLINE_API SimpleWriterRdmaOp : public Operator, public Metadata {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(SimpleWriterRdmaOp)

    ~SimpleWriterRdmaOp();

    void initialize() override;
    void setup(OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
    struct Impl;
    Impl* pimpl;

    Parameter<std::string> filePath_;
};

class STELLINE_API DummyWriterOp : public Operator, public Metadata {
 public:
      HOLOSCAN_OPERATOR_FORWARD_ARGS(DummyWriterOp)

      DummyWriterOp() = default;
      ~DummyWriterOp();

      void initialize() override;
      void setup(OperatorSpec& spec) override;
      void start() override;
      void stop() override;
      void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
      struct Impl;
      Impl* pimpl;
};

}  // namespace stelline::operators::filesystem

#endif  // STELLINE_OPERATORS_FILESYSTEM_BASE_HH
