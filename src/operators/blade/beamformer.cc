#include <blade/base.hh>
#include <blade/modules/beamformer/ata.hh>
#include <blade/modules/channelizer/base.hh>
#include <blade/modules/caster.hh>

#include <stelline/operators/blade/base.hh>
#include <stelline/types.hh>

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

        U64 inputCasterBlockSize = 512;
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

        this->connect(inputCaster, {
            .blockSize = config.inputCasterBlockSize,
        }, {
            .buf = inputBuffer,
        });

        this->connect(beamformer, {}, {
            .buf = inputCaster->getOutputBuffer(),
            .phasors = inputPhasorBuffer,
        });

        this->connect(channelizer, {
            .rate = 8192,
        }, {
            .buf = beamformer->getOutputBuffer(),
        });
    }

    Result transferIn(const ArrayTensor<Device::CUDA, IT>& deviceInputBuffer) {
        BL_CHECK(this->copy(inputBuffer, deviceInputBuffer));
        return Result::SUCCESS;
    }

    Result transferResult() {
        BL_CHECK(this->copy(outputBuffer, channelizer->getOutputBuffer()));
        return Result::SUCCESS;
    }

    Result transferOut(ArrayTensor<Device::CUDA, OT>& deviceOutputBuffer) {
        BL_CHECK(this->copy(deviceOutputBuffer, outputBuffer));
        return Result::SUCCESS;
    }

private:
    using InputCaster = typename Modules::Caster<IT, CF32>;
    std::shared_ptr<InputCaster> inputCaster;

    using Beamformer = typename Modules::Beamformer::ATA<CF32, CF32>;
    std::shared_ptr<Beamformer> beamformer;

    using Channelizer = typename Modules::Channelizer<CF32, CF32>;
    std::shared_ptr<Channelizer> channelizer;

    Duet<ArrayTensor<Device::CUDA, IT>> inputBuffer;
    Duet<ArrayTensor<Device::CUDA, OT>> outputBuffer;

    PhasorTensor<Device::CUDA, CF32> inputPhasorBuffer;
};

struct BeamformerOp::Impl {
    // Configuration parameters (derived).

    BlockShape inputShape;
    BlockShape outputShape;

    // State.

    Dispatcher dispatcher;
    std::shared_ptr<OpBeamformerPipeline> pipeline;
    ArrayShape bladeInputShape;
    ArrayShape bladeOutputShape;

    // Metrics.

    std::thread metricsThread;
    bool metricsThreadRunning;
    void metricsLoop();
};

void BeamformerOp::initialize() {
    // Register custom types.
    register_converter<BlockShape>();

    // Allocate memory.
    pimpl = new Impl();

    // Initialize operator.
    Operator::initialize();
}

BeamformerOp::~BeamformerOp() {
    delete pimpl;
}

void BeamformerOp::setup(OperatorSpec& spec) {
    spec.input<DspBlock>("dsp_block_in")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));
    spec.output<DspBlock>("dsp_block_out")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));

    spec.param(inputShape_, "input_shape");
    spec.param(outputShape_, "output_shape");
}

void BeamformerOp::start() {
    // Convert Parameters to variables.

    pimpl->inputShape = inputShape_.get();
    pimpl->outputShape = outputShape_.get();

    pimpl->bladeInputShape = ArrayShape({
        pimpl->inputShape.numberOfAntennas,
        pimpl->inputShape.numberOfChannels,
        pimpl->inputShape.numberOfSamples,
        pimpl->inputShape.numberOfPolarizations
    });

    pimpl->bladeOutputShape = ArrayShape({
        pimpl->outputShape.numberOfAntennas,
        pimpl->outputShape.numberOfChannels,
        pimpl->outputShape.numberOfSamples,
        pimpl->outputShape.numberOfPolarizations
    });

    // Validate configuration.

    // TODO: Write validation.

    // Create pipeline.
    // TODO: Implement configuration parameters.

    OpBeamformerPipeline::Config config = {
        .inputShape = pimpl->bladeInputShape,
        .outputShape = pimpl->bladeOutputShape
    };
    pimpl->pipeline = std::make_shared<OpBeamformerPipeline>(config);

    // Initialize Dispatcher.

    // TODO: Make number of buffers configurable.
    pimpl->dispatcher.template initialize<CF32>(8, pimpl->bladeOutputShape);

    // Start metrics thread.

    pimpl->metricsThreadRunning = true;
    pimpl->metricsThread = std::thread([&]{
        pimpl->metricsLoop();
    });
}

void BeamformerOp::stop() {
    // Stop metrics thread.

    pimpl->metricsThreadRunning = false;
    if (pimpl->metricsThread.joinable()) {
        pimpl->metricsThread.join();
    }
}

void BeamformerOp::compute(InputContext& input, OutputContext& output, ExecutionContext&) {
    auto receiveCallback = [&](){
        return input.receive<DspBlock>("dsp_block_in").value();
    };

    auto convertInputCallback = [&](DspBlock& data){
        ArrayTensor<Device::CUDA, CF32> deviceInputBuffer(data.tensor->data(), pimpl->bladeInputShape);
        return pimpl->pipeline->transferIn(deviceInputBuffer);
    };

    auto convertOutputCallback = [&](DspBlock& data){
        ArrayTensor<Device::CUDA, CF32> deviceOutputBuffer(data.tensor->data(), pimpl->bladeOutputShape);
        return pimpl->pipeline->transferOut(deviceOutputBuffer);
    };

    auto emitCallback = [&](DspBlock& data){
        output.emit(data, "dsp_block_out");
    };

    if (pimpl->dispatcher.run(pimpl->pipeline,
                              receiveCallback,
                              convertInputCallback,
                              convertOutputCallback,
                              emitCallback) != Result::SUCCESS) {
        throw std::runtime_error("Dispatcher failed.");
    }
}

void BeamformerOp::Impl::metricsLoop() {
    while (metricsThreadRunning) {
        HOLOSCAN_LOG_INFO("Beamformer Operator:");
        dispatcher.metrics();

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

}  // namespace stelline::operators::blade
