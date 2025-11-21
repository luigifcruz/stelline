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

class OpCorrelatorPipeline : public Blade::Runner {
public:
    using IT = CF32;
    using OT = CF32;

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

struct CorrelatorOp::Impl {
    // Derived configuration parameters.

    uint64_t numberOfBuffers;
    BlockShape inputShape;
    BlockShape outputShape;
    Map options;

    // State.

    Dispatcher dispatcher;
    OpCorrelatorPipeline::Config config;
    std::shared_ptr<OpCorrelatorPipeline> pipeline;
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
    spec.input<std::shared_ptr<holoscan::Tensor>>("dsp_block_in")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));
    spec.output<std::shared_ptr<holoscan::Tensor>>("dsp_block_out")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));

    spec.param(numberOfBuffers_, "number_of_buffers");
    spec.param(inputShape_, "input_shape");
    spec.param(outputShape_, "output_shape");
    spec.param(options_, "options");
}

void CorrelatorOp::start() {
    // Convert Parameters to variables.

    pimpl->numberOfBuffers = numberOfBuffers_.get();
    pimpl->inputShape = inputShape_.get();
    pimpl->outputShape = outputShape_.get();
    pimpl->options = options_.get();

    // Validate configuration.

    // TODO: Write validation.

    // Create pipeline.

    pimpl->config = {
        .inputShape = ArrayShape({
            pimpl->inputShape.numberOfAntennas,
            pimpl->inputShape.numberOfChannels,
            pimpl->inputShape.numberOfSamples,
            pimpl->inputShape.numberOfPolarizations
        }),
        .outputShape = ArrayShape({
            pimpl->outputShape.numberOfAntennas,
            pimpl->outputShape.numberOfChannels,
            pimpl->outputShape.numberOfSamples,
            pimpl->outputShape.numberOfPolarizations
        }),

        .preChannelizerStackerMultiplier = FetchMap<U64>(pimpl->options, "pre_channelizer_stacker_multiplier", 1),

        .channelizerBypass = FetchMap<bool>(pimpl->options, "channelizer_bypass", false),

        .preCorrelatorStackerMultiplier = FetchMap<U64>(pimpl->options, "pre_correlator_stacker_multiplier", 8),

        .correlatorIntegrationRate = FetchMap<U64>(pimpl->options, "correlator_integration_rate", 128),
        .correlatorConjugateAntennaIndex = FetchMap<U64>(pimpl->options, "correlator_conjugate_antenna_index", 1),
        .correlatorUseSharedMemory = FetchMap<bool>(pimpl->options, "correlator_use_shared_memory", false),
        .correlatorCalculationMode = CALC_MODE::INTEGER,

        .stackerBlockSize = FetchMap<U64>(pimpl->options, "stacker_block_size", 512),
        .casterBlockSize = FetchMap<U64>(pimpl->options, "caster_block_size", 512),
        .channelizerBlockSize = FetchMap<U64>(pimpl->options, "channelizer_block_size", 512),
        .correlatorBlockSize = FetchMap<U64>(pimpl->options, "correlator_block_size", 64),
    };
    pimpl->pipeline = std::make_shared<OpCorrelatorPipeline>(pimpl->config);

    // Initialize Dispatcher.

    // TODO: Make number of buffers configurable.
    pimpl->dispatcher.template initialize<CF32>(pimpl->numberOfBuffers, pimpl->config.outputShape);
}

void CorrelatorOp::stop() {
}

void CorrelatorOp::compute(InputContext& input, OutputContext& output, ExecutionContext&) {
    auto receiveCallback = [&](){
        return input.receive<std::shared_ptr<holoscan::Tensor>>("dsp_block_in").value();
    };

    auto convertInputCallback = [&](std::shared_ptr<holoscan::Tensor>& tensor){
        ArrayTensor<Device::CUDA, CF32> deviceInputBuffer(tensor->data(), pimpl->config.inputShape);
        return pimpl->pipeline->transferIn(deviceInputBuffer);
    };

    auto convertOutputCallback = [&](std::shared_ptr<holoscan::Tensor>& tensor){
        ArrayTensor<Device::CUDA, CF32> deviceOutputBuffer(tensor->data(), pimpl->config.outputShape);
        return pimpl->pipeline->transferOut(deviceOutputBuffer);
    };

    auto emitCallback = [&](std::shared_ptr<holoscan::Tensor>& tensor){
        output.emit(tensor, "dsp_block_out");
    };

    if (pimpl->dispatcher.run(pimpl->pipeline,
                              receiveCallback,
                              convertInputCallback,
                              convertOutputCallback,
                              emitCallback,
                              metadata()) != Result::SUCCESS) {
        throw std::runtime_error("Dispatcher failed.");
    }
}

stelline::StoreInterface::MetricsMap CorrelatorOp::collectMetricsMap() {
    const auto stats = pimpl->dispatcher.metrics();
    stelline::StoreInterface::MetricsMap metrics;
    metrics["successful_enqueues"] = fmt::format("{}", stats.successfulEnqueues);
    metrics["successful_dequeues"] = fmt::format("{}", stats.successfulDequeues);
    metrics["full_enqueues"] = fmt::format("{}", stats.fullEnqueues);
    metrics["dequeue_retries"] = fmt::format("{}", stats.dequeueRetries);
    metrics["premature_dequeues"] = fmt::format("{}", stats.prematureDequeues);
    return metrics;
}

std::string CorrelatorOp::collectMetricsString() {
    const auto metrics = collectMetricsMap();
    return fmt::format(
        "Correlator Operator:\n"
        "  Queueing Statistics:\n"
        "    Successful Enqueues: {}\n"
        "    Successful Dequeues: {}\n"
        "    Full Enqueues: {}\n"
        "    Dequeue Retries: {}\n"
        "    Premature Dequeues: {}",
        metrics.at("successful_enqueues"),
        metrics.at("successful_dequeues"),
        metrics.at("full_enqueues"),
        metrics.at("dequeue_retries"),
        metrics.at("premature_dequeues"));
}

}  // namespace stelline::operators::blade
