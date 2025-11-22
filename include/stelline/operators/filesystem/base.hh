#ifndef STELLINE_OPERATORS_FILESYSTEM_BASE_HH
#define STELLINE_OPERATORS_FILESYSTEM_BASE_HH

#include <holoscan/holoscan.hpp>

#include <stelline/common.hh>
#include <stelline/store.hh>


namespace stelline::operators::filesystem {

using holoscan::Operator;
using holoscan::Parameter;
using holoscan::OperatorSpec;
using holoscan::InputContext;
using holoscan::OutputContext;
using holoscan::ExecutionContext;
class STELLINE_API SimpleWriterOp : public Operator, public stelline::StoreInterface {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(SimpleWriterOp)

    ~SimpleWriterOp();

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

    Parameter<std::string> filePath_;
};

class STELLINE_API SimpleWriterRdmaOp : public Operator, public stelline::StoreInterface {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(SimpleWriterRdmaOp)

    ~SimpleWriterRdmaOp();

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

    Parameter<std::string> filePath_;
};

class STELLINE_API DummyWriterOp : public Operator, public stelline::StoreInterface {
 public:
      HOLOSCAN_OPERATOR_FORWARD_ARGS(DummyWriterOp)

      DummyWriterOp() = default;
      ~DummyWriterOp();

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
};

#ifdef STELLINE_LOADER_FBH5
class STELLINE_API Fbh5WriterRdmaOp : public Operator, public stelline::StoreInterface {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(Fbh5WriterRdmaOp)

    ~Fbh5WriterRdmaOp();

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

    Parameter<std::string> filePath_;
};
#endif  // STELLINE_LOADER_FBH5

#ifdef STELLINE_LOADER_UVH5
class STELLINE_API Uvh5WriterRdmaOp : public Operator, public stelline::StoreInterface {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(Uvh5WriterRdmaOp)

    ~Uvh5WriterRdmaOp();

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

    Parameter<std::string> output_filePath_;
    Parameter<std::string> telinfo_filePath_;
    Parameter<std::string> obsantinfo_filePath_;
    Parameter<std::string> iers_filePath_;
};
#endif  // STELLINE_LOADER_UVH5

}  // namespace stelline::operators::filesystem

#endif  // STELLINE_OPERATORS_FILESYSTEM_BASE_HH
