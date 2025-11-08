#ifndef STELLINE_OPERATORS_BLADE_BASE_HH
#define STELLINE_OPERATORS_BLADE_BASE_HH

#include <holoscan/holoscan.hpp>

#include <stelline/common.hh>

#include <stelline/yaml/types/block_shape.hh>
#include <stelline/yaml/types/map.hh>

namespace stelline::operators::blade {

using holoscan::Operator;
using holoscan::Parameter;
using holoscan::OperatorSpec;
using holoscan::InputContext;
using holoscan::OutputContext;
using holoscan::ExecutionContext;

class STELLINE_API CorrelatorOp : public Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(CorrelatorOp)

    ~CorrelatorOp();

    void initialize() override;
    void setup(OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
    struct Impl;
    Impl* pimpl;

    Parameter<uint64_t> numberOfBuffers_;
    Parameter<BlockShape> inputShape_;
    Parameter<BlockShape> outputShape_;
    Parameter<Map> options_;
};

class STELLINE_API FrbnnOp : public Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(FrbnnOp)

    ~FrbnnOp();

    void initialize() override;
    void setup(OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
    struct Impl;
    Impl* pimpl;

    Parameter<uint64_t> numberOfBuffers_;
    Parameter<BlockShape> inputShape_;
    Parameter<BlockShape> outputShape_;
    Parameter<Map> options_;
};

class STELLINE_API BeamformerOp : public Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(BeamformerOp)

    ~BeamformerOp();

    void initialize() override;
    void setup(OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
    struct Impl;
    Impl* pimpl;

    Parameter<uint64_t> numberOfBuffers_;
    Parameter<BlockShape> inputShape_;
    Parameter<BlockShape> outputShape_;
    Parameter<Map> options_;
};

}  // namespace stelline::operators::blade

#endif  // STELLINE_OPERATORS_BLADE_BASE_HH
