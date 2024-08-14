#ifndef CYBERBRIDGE_HOLOSCAN_SOURCE_MODULE_HH
#define CYBERBRIDGE_HOLOSCAN_SOURCE_MODULE_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"
#include "jetstream/memory/utils/tensor_circular_buffer.hh"

namespace Jetstream {

#define JST_SOURCE_CPU(MACRO) \
    MACRO(Source, CPU, CF32) \
    MACRO(Source, CPU, F32) \
    MACRO(Source, CPU, CI8)

#define JST_SOURCE_CUDA(MACRO) \
    MACRO(Source, CUDA, CF32) \
    MACRO(Source, CUDA, F32) \
    MACRO(Source, CUDA, CI8)

template<Device D, typename T = CF32>
class Source : public Module, public Compute {
 public:
    // Configuration 

    struct Config {
        std::string nodeName;
        std::string nodeOutputName;
        std::vector<U64> shape = {2, 8192};

        JST_SERDES(nodeName, nodeOutputName, shape);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        JST_SERDES_INPUT();
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, T> buffer;

        JST_SERDES_OUTPUT(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, T>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    void info() const final;

    const std::string& warning() const;

    // Constructor

    Result create();
    Result destroy();

 protected:
    Result computeReady() final;
    Result compute(const Context& ctx) final;

    class Op : public holoscan::Operator {
     public:
        HOLOSCAN_OPERATOR_FORWARD_ARGS(Op)

        Op() = default;

        void setup(holoscan::OperatorSpec& spec) override;

        void start() override;

        void compute(holoscan::InputContext& input, 
                     holoscan::OutputContext&, 
                     holoscan::ExecutionContext&) override;

        const std::optional<std::string>& status() const;
        std::vector<U64>& shape();

     protected:
        Memory::TensorCircularBuffer<Tensor<D, T>> buffer;
        std::optional<std::string> _status;
        std::vector<U64> _shape;
    
        friend class Source;
    };

 private:
    JST_DEFINE_IO();

    std::string holoscanModuleId;
    std::shared_ptr<Op> op;
    std::string _warning;
};

JST_SOURCE_CPU(JST_SPECIALIZATION);
JST_SOURCE_CUDA(JST_SPECIALIZATION);

}  // namespace Jetstream

#endif  // CYBERBRIDGE_HOLOSCAN_SOURCE_MODULE_HH