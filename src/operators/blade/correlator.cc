#include <blade/base.hh>
#include <blade/bundles/generic/mode_x.hh>

#include <stelline/operators/blade/base.hh>
#include <stelline/types.hh>
#include <fmt/format.h>

#include "utils/dispatcher.hh"

using namespace Blade;
using namespace holoscan;
using namespace holoscan::ops;

namespace stelline::operators::blade {

template<typename IT, typename OT>
class OpCorrelatorPipeline : public Blade::Runner {
public:
    using ModeX = Bundles::Generic::ModeX<IT, OT>;
    using Config = ModeX::Config;

    explicit OpCorrelatorPipeline(const Config& config)
            : inputBuffer(config.inputShape),
            outputBuffer(config.outputShape) {
        this->connect(modeX, config, {
            .buffer = inputBuffer,
        });
    }

    Result transferIn(const ArrayTensor<Device::CUDA, IT>& deviceInputBuffer) {
        BL_CHECK(this->copy(inputBuffer, deviceInputBuffer));
        return Result::SUCCESS;
    }

    Result transferResult() {
        BL_CHECK(this->copy(outputBuffer, modeX->getOutputBuffer()));
        return Result::SUCCESS;
    }

    Result transferOut(ArrayTensor<Device::CUDA, OT>& deviceOutputBuffer) {
        BL_CHECK(this->copy(deviceOutputBuffer, outputBuffer));
        return Result::SUCCESS;
    }

private:
    std::shared_ptr<ModeX> modeX;

    Duet<ArrayTensor<Device::CUDA, IT>> inputBuffer;
    Duet<ArrayTensor<Device::CUDA, OT>> outputBuffer;
};

using OpCorrelatorPipelineCF32 = OpCorrelatorPipeline<CF32, CF32>;
using OpCorrelatorPipelineCI8 = OpCorrelatorPipeline<CI8, CF32>;

struct CorrelatorOp::Impl {
    // Derived configuration parameters.

    uint64_t numberOfBuffers;
    BlockShape inputShape;
    BlockShape outputShape;
    Map options;
    std::string dtype;

    // State.

    Dispatcher dispatcher;
    OpCorrelatorPipelineCF32::Config configCF32;
    OpCorrelatorPipelineCI8::Config configCI8;
    std::shared_ptr<OpCorrelatorPipelineCF32> pipelineCF32;
    std::shared_ptr<OpCorrelatorPipelineCI8> pipelineCI8;
};

void CorrelatorOp::initialize() {
    // Register custom types.
    register_converter<BlockShape>();
    register_converter<Map>();

    // Allocate memory.
    pimpl = new Impl();

    // Initialize operator.
    Operator::initialize();
}

CorrelatorOp::~CorrelatorOp() {
    delete pimpl;
}

void CorrelatorOp::setup(OperatorSpec& spec) {
    spec.input<std::shared_ptr<holoscan::Tensor>>("in")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));
    spec.output<std::shared_ptr<holoscan::Tensor>>("out")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));

    spec.param(numberOfBuffers_, "number_of_buffers");
    spec.param(inputShape_, "input_shape");
    spec.param(outputShape_, "output_shape");
    spec.param(options_, "options");
    spec.param(dtype_, "dtype");
}

void CorrelatorOp::start() {
    // Convert Parameters to variables.

    pimpl->numberOfBuffers = numberOfBuffers_.get();
    pimpl->inputShape = inputShape_.get();
    pimpl->outputShape = outputShape_.get();
    pimpl->options = options_.get();
    pimpl->dtype = dtype_.get();

    // Validate configuration.

    // TODO: Write validation.

    // Create pipeline.

    if (pimpl->dtype == "ci8") {
        pimpl->configCI8.inputShape = ArrayShape({
            pimpl->inputShape.numberOfAntennas,
            pimpl->inputShape.numberOfChannels,
            pimpl->inputShape.numberOfSamples,
            pimpl->inputShape.numberOfPolarizations
        });
        pimpl->configCI8.outputShape = ArrayShape({
            pimpl->outputShape.numberOfAntennas,
            pimpl->outputShape.numberOfChannels,
            pimpl->outputShape.numberOfSamples,
            pimpl->outputShape.numberOfPolarizations
        });

        pimpl->configCI8.preChannelizerStackerMultiplier = FetchMap<U64>(pimpl->options, "pre_channelizer_stacker_multiplier", 1);
        pimpl->configCI8.channelizerBypass = FetchMap<bool>(pimpl->options, "channelizer_bypass", false);
        pimpl->configCI8.preCorrelatorStackerMultiplier = FetchMap<U64>(pimpl->options, "pre_correlator_stacker_multiplier", 8);
        pimpl->configCI8.correlatorIntegrationRate = FetchMap<U64>(pimpl->options, "correlator_integration_rate", 128);
        pimpl->configCI8.correlatorConjugateAntennaIndex = FetchMap<U64>(pimpl->options, "correlator_conjugate_antenna_index", 1);
        pimpl->configCI8.correlatorUseSharedMemory = FetchMap<bool>(pimpl->options, "correlator_use_shared_memory", false);
        pimpl->configCI8.correlatorCalculationMode = CALC_MODE::INTEGER;
        pimpl->configCI8.postCorrelatorFrequencyIntegrationRate = FetchMap<U64>(pimpl->options, "post_correlator_frequency_integration_rate", 1);
        pimpl->configCI8.stackerBlockSize = FetchMap<U64>(pimpl->options, "stacker_block_size", 512);
        pimpl->configCI8.casterBlockSize = FetchMap<U64>(pimpl->options, "caster_block_size", 512);
        pimpl->configCI8.channelizerBlockSize = FetchMap<U64>(pimpl->options, "channelizer_block_size", 512);
        pimpl->configCI8.correlatorBlockSize = FetchMap<U64>(pimpl->options, "correlator_block_size", 64);

        pimpl->pipelineCI8 = std::make_shared<OpCorrelatorPipelineCI8>(pimpl->configCI8);
        pimpl->dispatcher.template initialize<CF32>(pimpl->numberOfBuffers, pimpl->configCI8.outputShape);
    }

    if (pimpl->dtype == "cf32") {
        pimpl->configCF32.inputShape = ArrayShape({
            pimpl->inputShape.numberOfAntennas,
            pimpl->inputShape.numberOfChannels,
            pimpl->inputShape.numberOfSamples,
            pimpl->inputShape.numberOfPolarizations
        });
        pimpl->configCF32.outputShape = ArrayShape({
            pimpl->outputShape.numberOfAntennas,
            pimpl->outputShape.numberOfChannels,
            pimpl->outputShape.numberOfSamples,
            pimpl->outputShape.numberOfPolarizations
        });
        pimpl->configCF32.preChannelizerStackerMultiplier = FetchMap<U64>(pimpl->options, "pre_channelizer_stacker_multiplier", 1);
        pimpl->configCF32.channelizerBypass = FetchMap<bool>(pimpl->options, "channelizer_bypass", false);
        pimpl->configCF32.preCorrelatorStackerMultiplier = FetchMap<U64>(pimpl->options, "pre_correlator_stacker_multiplier", 8);
        pimpl->configCF32.correlatorIntegrationRate = FetchMap<U64>(pimpl->options, "correlator_integration_rate", 128);
        pimpl->configCF32.correlatorConjugateAntennaIndex = FetchMap<U64>(pimpl->options, "correlator_conjugate_antenna_index", 1);
        pimpl->configCF32.correlatorUseSharedMemory = FetchMap<bool>(pimpl->options, "correlator_use_shared_memory", false);
        pimpl->configCF32.correlatorCalculationMode = CALC_MODE::INTEGER;
        pimpl->configCF32.postCorrelatorFrequencyIntegrationRate = FetchMap<U64>(pimpl->options, "post_correlator_frequency_integration_rate", 1);
        pimpl->configCF32.stackerBlockSize = FetchMap<U64>(pimpl->options, "stacker_block_size", 512);
        pimpl->configCF32.casterBlockSize = FetchMap<U64>(pimpl->options, "caster_block_size", 512);
        pimpl->configCF32.channelizerBlockSize = FetchMap<U64>(pimpl->options, "channelizer_block_size", 512);
        pimpl->configCF32.correlatorBlockSize = FetchMap<U64>(pimpl->options, "correlator_block_size", 64);

        pimpl->pipelineCF32 = std::make_shared<OpCorrelatorPipelineCF32>(pimpl->configCF32);
        pimpl->dispatcher.template initialize<CF32>(pimpl->numberOfBuffers, pimpl->configCF32.outputShape);
    }
}

void CorrelatorOp::stop() {
}

void CorrelatorOp::compute(InputContext& input, OutputContext& output, ExecutionContext&) {
    auto receiveCallback = [&](){
        auto result = input.receive<std::shared_ptr<holoscan::Tensor>>("in");
        if (!result) {
            throw std::runtime_error("No input tensor available.");
        }

        return result.value();
    };

    auto convertInputCallback = [&](std::shared_ptr<holoscan::Tensor>& tensor){
        if (pimpl->dtype == "ci8") {
            ArrayTensor<Device::CUDA, CI8> deviceInputBuffer(tensor->data(), pimpl->configCI8.inputShape);
            return pimpl->pipelineCI8->transferIn(deviceInputBuffer);
        }
        if (pimpl->dtype == "cf32") {
            ArrayTensor<Device::CUDA, CF32> deviceInputBuffer(tensor->data(), pimpl->configCF32.inputShape);
            return pimpl->pipelineCF32->transferIn(deviceInputBuffer);
        }
        throw std::runtime_error("Unsupported data type");
    };

    auto convertOutputCallback = [&](std::shared_ptr<holoscan::Tensor>& tensor){
        if (pimpl->dtype == "ci8") {
            ArrayTensor<Device::CUDA, CF32> deviceOutputBuffer(tensor->data(), pimpl->configCI8.outputShape);
            return pimpl->pipelineCI8->transferOut(deviceOutputBuffer);
        }
        if (pimpl->dtype == "cf32") {
            ArrayTensor<Device::CUDA, CF32> deviceOutputBuffer(tensor->data(), pimpl->configCF32.outputShape);
            return pimpl->pipelineCF32->transferOut(deviceOutputBuffer);
        }
        throw std::runtime_error("Unsupported data type");
    };

    auto emitCallback = [&](std::shared_ptr<holoscan::Tensor>& tensor){
        output.emit(tensor, "out");
    };

    if (pimpl->dtype == "ci8") {
        if (pimpl->dispatcher.run(pimpl->pipelineCI8,
                                  receiveCallback,
                                  convertInputCallback,
                                  convertOutputCallback,
                                  emitCallback,
                                  metadata()) != Result::SUCCESS) {
            throw std::runtime_error("Dispatcher failed.");
        }
    }

    if (pimpl->dtype == "cf32") {
        if (pimpl->dispatcher.run(pimpl->pipelineCF32,
                                  receiveCallback,
                                  convertInputCallback,
                                  convertOutputCallback,
                                  emitCallback,
                                  metadata()) != Result::SUCCESS) {
            throw std::runtime_error("Dispatcher failed.");
        }
    }
}

void CorrelatorOp::tick() {
    if (!pimpl || !metrics()) {
        return;
    }
    const auto stats = pimpl->dispatcher.metrics();
    metrics()->record("successful_enqueues", fmt::format("{}", stats.successfulEnqueues));
    metrics()->record("successful_dequeues", fmt::format("{}", stats.successfulDequeues));
    metrics()->record("full_enqueues", fmt::format("{}", stats.fullEnqueues));
    metrics()->record("dequeue_retries", fmt::format("{}", stats.dequeueRetries));
    metrics()->record("premature_dequeues", fmt::format("{}", stats.prematureDequeues));
}

std::string CorrelatorOp::formatMetrics(const MetricsProvider::MetricsMap& metrics) {
    return fmt::format("  Queueing Statistics:\n"
                       "    Successful Enqueues: {}\n"
                       "    Successful Dequeues: {}\n"
                       "    Full Enqueues: {}\n"
                       "    Dequeue Retries: {}\n"
                       "    Premature Dequeues: {}",
                       metrics.at("successful_enqueues").value,
                       metrics.at("successful_dequeues").value,
                       metrics.at("full_enqueues").value,
                       metrics.at("dequeue_retries").value,
                       metrics.at("premature_dequeues").value);
}

}  // namespace stelline::operators::blade
