#include <blade/base.hh>
#include <blade/modules/beamformer/ata.hh>
#include <blade/modules/channelizer/base.hh>
#include <blade/modules/caster.hh>

#include <stelline/operators/blade/base.hh>
#include <stelline/types.hh>
#include <fmt/format.h>

#include "utils/dispatcher.hh"

using namespace Blade;
using namespace holoscan;
using namespace holoscan::ops;

namespace stelline::operators::blade {

class OpBeamformerPipeline : public Blade::Runner {
public:
    using IT = CF32;
    using OT = CF32;

    struct Config {
        ArrayShape inputShape;
        ArrayShape outputShape;

        bool beamformerEnableIncoherentBeam;
        bool beamformerEnableIncoherentBeamSqrt;

        U64 beamformerBlockSize;
    };

    explicit OpBeamformerPipeline(const Config& config)
            : inputBuffer(config.inputShape),
              outputBuffer(config.outputShape) {
        // Load input phasor buffer with 1.0+0.0i.

        auto inputPhasorShape = PhasorShape({
            1,
            inputBuffer[0].shape().numberOfAspects(),
            inputBuffer[0].shape().numberOfFrequencyChannels(),
            inputBuffer[0].shape().numberOfTimeSamples(),
            inputBuffer[0].shape().numberOfPolarizations()
        });

        inputPhasorBuffer = PhasorTensor<Device::CUDA, CF32>(inputPhasorShape);
        auto hostInputPhasorBuffer = PhasorTensor<Device::CPU, CF32>(inputPhasorBuffer.shape());

        for (int i = 0; i < hostInputPhasorBuffer.size(); i++) {
            hostInputPhasorBuffer[i] = std::complex<float>(1.0, 0.0);
        }

        Copy(inputPhasorBuffer, hostInputPhasorBuffer);

        // Connect modules.

        this->connect(beamformer, {
            .enableIncoherentBeam = config.beamformerEnableIncoherentBeam,
            .enableIncoherentBeamSqrt = config.beamformerEnableIncoherentBeamSqrt,

            .blockSize = config.beamformerBlockSize,
        }, {
            .buf = inputBuffer,
            .phasors = inputPhasorBuffer,
        });
    }

    Result transferIn(const ArrayTensor<Device::CUDA, IT>& deviceInputBuffer) {
        BL_CHECK(this->copy(inputBuffer, deviceInputBuffer));
        return Result::SUCCESS;
    }

    Result transferResult() {
        BL_CHECK(this->copy(outputBuffer, beamformer->getOutputBuffer()));
        return Result::SUCCESS;
    }

    Result transferOut(ArrayTensor<Device::CUDA, OT>& deviceOutputBuffer) {
        BL_CHECK(this->copy(deviceOutputBuffer, outputBuffer));
        return Result::SUCCESS;
    }

private:
    using Beamformer = typename Modules::Beamformer::ATA<CF32, CF32>;
    std::shared_ptr<Beamformer> beamformer;

    Duet<ArrayTensor<Device::CUDA, IT>> inputBuffer;
    Duet<ArrayTensor<Device::CUDA, OT>> outputBuffer;

    PhasorTensor<Device::CUDA, CF32> inputPhasorBuffer;
};

struct BeamformerOp::Impl {
    // Configuration parameters (derived).

    uint64_t numberOfBuffers;
    BlockShape inputShape;
    BlockShape outputShape;
    Map options;

    // State.

    Dispatcher dispatcher;
    OpBeamformerPipeline::Config config;
    std::shared_ptr<OpBeamformerPipeline> pipeline;
};

void BeamformerOp::initialize() {
    // Register custom types.
    register_converter<BlockShape>();
    register_converter<Map>();

    // Allocate memory.
    pimpl = new Impl();

    // Initialize operator.
    Operator::initialize();
}

BeamformerOp::~BeamformerOp() {
    delete pimpl;
}

void BeamformerOp::setup(OperatorSpec& spec) {
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

void BeamformerOp::start() {
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

        .beamformerEnableIncoherentBeam = FetchMap<bool>(pimpl->options, "beamformer_enable_incoherent_beam", false),
        .beamformerEnableIncoherentBeamSqrt = FetchMap<bool>(pimpl->options, "beamformer_enable_incoherent_beam_sqrt", false),

        .beamformerBlockSize = FetchMap<U64>(pimpl->options, "beamformer_block_size", 512),
    };
    pimpl->pipeline = std::make_shared<OpBeamformerPipeline>(pimpl->config);

    // Initialize Dispatcher.

    // TODO: Make number of buffers configurable.
    pimpl->dispatcher.template initialize<CF32>(pimpl->numberOfBuffers, pimpl->config.outputShape);
}

void BeamformerOp::stop() {
}

void BeamformerOp::compute(InputContext& input, OutputContext& output, ExecutionContext&) {
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

stelline::StoreInterface::MetricsMap BeamformerOp::collectMetricsMap() {
    const auto stats = pimpl->dispatcher.metrics();
    stelline::StoreInterface::MetricsMap metrics;
    metrics["successful_enqueues"] = fmt::format("{}", stats.successfulEnqueues);
    metrics["successful_dequeues"] = fmt::format("{}", stats.successfulDequeues);
    metrics["full_enqueues"] = fmt::format("{}", stats.fullEnqueues);
    metrics["dequeue_retries"] = fmt::format("{}", stats.dequeueRetries);
    metrics["premature_dequeues"] = fmt::format("{}", stats.prematureDequeues);
    return metrics;
}

std::string BeamformerOp::collectMetricsString() {
    const auto metrics = collectMetricsMap();
    return fmt::format(
        "Beamformer Operator:\n"
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
