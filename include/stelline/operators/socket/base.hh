#ifndef STELLINE_OPERATORS_SOCKET_BASE_HH
#define STELLINE_OPERATORS_SOCKET_BASE_HH

#include <holoscan/holoscan.hpp>

#include <stelline/common.hh>
#include <stelline/metadata.hh>

namespace stelline::operators::socket {

using holoscan::Operator;
using holoscan::Parameter;
using holoscan::OperatorSpec;
using holoscan::InputContext;
using holoscan::OutputContext;
using holoscan::ExecutionContext;

class STELLINE_API ZmqTransmitterOp : public Operator, public Metadata {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(ZmqTransmitterOp)

    ~ZmqTransmitterOp();

    void initialize() override;
    void setup(OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
    struct Impl;
    Impl* pimpl;

    Parameter<std::string> address_;
};

}  // namespace stelline::operators::socket

#endif  // STELLINE_OPERATORS_SOCKET_BASE_HH
