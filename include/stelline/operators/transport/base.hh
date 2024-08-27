#ifndef STELLINE_OPERATORS_TRANSPORT_BASE_HH
#define STELLINE_OPERATORS_TRANSPORT_BASE_HH

#include <holoscan/holoscan.hpp>

#include <stelline/common.hh>
#include <stelline/yaml/block_shape.hh>

namespace stelline::operators::transport {

using holoscan::Operator;
using holoscan::Parameter;
using holoscan::OperatorSpec;
using holoscan::InputContext;
using holoscan::OutputContext;
using holoscan::ExecutionContext;

class STELLINE_API ReceiverOp : public Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(ReceiverOp)

    constexpr static const uint64_t TransportHeaderSize = 46;
    constexpr static const uint64_t VoltageHeaderSize = 16;
    constexpr static const uint64_t VoltageDataSize = 6144;

    ~ReceiverOp();

    void initialize() override;
    void setup(OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
    struct Impl;
    Impl* pimpl;

    Parameter<BlockShape> totalBlock_;
    Parameter<BlockShape> partialBlock_;
    Parameter<BlockShape> offsetBlock_;
    Parameter<uint64_t> concurrentBlocks_;
};

}  // namespace stelline::operators::transport

#endif  // STELLINE_OPERATORS_TRANSPORT_BASE_HH
