#ifndef STELLINE_OPERATORS_TRANSPORT_BASE_HH
#define STELLINE_OPERATORS_TRANSPORT_BASE_HH

#include <holoscan/holoscan.hpp>

#include <stelline/common.hh>

#include <stelline/yaml/types/block_shape.hh>
#include <stelline/store.hh>

namespace stelline::operators::transport {

using holoscan::Operator;
using holoscan::Parameter;
using holoscan::OperatorSpec;
using holoscan::InputContext;
using holoscan::OutputContext;
using holoscan::ExecutionContext;

class STELLINE_API AtaReceiverOp : public Operator, public stelline::StoreInterface {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(AtaReceiverOp)

    ~AtaReceiverOp();

    void initialize() override;
    void setup(OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

    StoreInterface::MetricsMap collectMetricsMap() override;
    std::string collectMetricsString() override;

 private:
    struct Impl;
    Impl* pimpl;

    Parameter<BlockShape> totalBlock_;
    Parameter<BlockShape> partialBlock_;
    Parameter<BlockShape> offsetBlock_;
    Parameter<uint64_t> maxConcurrentBlocks_;
    Parameter<uint64_t> packetHeaderSize_;
    Parameter<uint64_t> packetHeaderOffset_;
    Parameter<uint64_t> outputPoolSize_;
    Parameter<bool> enableCsvLogging_;
};

class STELLINE_API DummyReceiverOp : public Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(DummyReceiverOp)

    ~DummyReceiverOp();

    void initialize() override;
    void setup(OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
    struct Impl;
    Impl* pimpl;

    Parameter<BlockShape> totalBlock_;
};

class STELLINE_API SorterOp : public Operator, public stelline::StoreInterface {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(SorterOp)

    ~SorterOp();

    void initialize() override;
    void setup(OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

    StoreInterface::MetricsMap collectMetricsMap() override;
    std::string collectMetricsString() override;

 private:
    struct Impl;
    Impl* pimpl;

    Parameter<uint64_t> depth_;
};

}  // namespace stelline::operators::transport

#endif  // STELLINE_OPERATORS_TRANSPORT_BASE_HH
