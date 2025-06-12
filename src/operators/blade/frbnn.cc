#include <blade/base.hh>
#include <blade/modules/beamformer/ata.hh>
#include <blade/modules/integrator.hh>
#include <blade/modules/stacker.hh>
#include <blade/modules/caster.hh>
#include <blade/modules/detector.hh>

#include <stelline/operators/blade/base.hh>
#include <stelline/types.hh>

#include "utils/dispatcher.hh"

using namespace Blade;
using namespace holoscan;
using namespace holoscan::ops;

namespace stelline::operators::blade {

class OpFrbnnPipeline : public Blade::Runner {
public:
    using IT = CF32;
    using OT = F32;

    struct Config {
        ArrayShape inputShape;
        ArrayShape outputShape;

        U64 integratorSize;
        U64 integratorRate;

        U64 timeStackerAxis;
        U64 timeStackerMultiplier;

        U64 batchStackerAxis;
        U64 batchStackerMultiplier;

        U64 inputCasterBlockSize;
        U64 integratorBlockSize;
        U64 detectorBlockSize;
        U64 outputCasterBlockSize;
        U64 batchStackerBlockSize;
        U64 timeStackerBlockSize;
    };

    explicit OpFrbnnPipeline(const Config& config)
            : inputBuffer(config.inputShape),
              outputBuffer(config.outputShape) {
        // Load input phasor buffer with 1.0+0.0i.

        auto inputPhasorShape = PhasorShape({
            1,
            inputBuffer.at(0).shape().numberOfAspects(),
            inputBuffer.at(0).shape().numberOfFrequencyChannels(),
            inputBuffer.at(0).shape().numberOfTimeSamples(),
            inputBuffer.at(0).shape().numberOfPolarizations()
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

        this->connect(detector, {
            .integrationRate = 1,
            .numberOfOutputPolarizations = 1,

            .blockSize = config.detectorBlockSize,
        }, {
            .buf = beamformer->getOutputBuffer(),
        });

        this->connect(integrator, {
            .size = config.integratorSize,
            .rate = config.integratorRate,

            .blockSize = config.integratorBlockSize,
        }, {
            .buf = detector->getOutputBuffer(),
        });

        this->connect(outputCaster, {
            .blockSize = config.outputCasterBlockSize,
        }, {
            .buf = integrator->getOutputBuffer(),
        });

        this->connect(timeStacker, {
            .axis = config.timeStackerAxis,
            .multiplier = config.timeStackerMultiplier,
            .copySizeThreshold = 256,

            .blockSize = config.timeStackerBlockSize,
        }, {
            .buf = outputCaster->getOutputBuffer(),
        });

        this->connect(batchStacker, {
            .axis = config.batchStackerAxis,
            .multiplier = config.batchStackerMultiplier,

            .blockSize = config.batchStackerBlockSize,
        }, {
            .buf = timeStacker->getOutputBuffer(),
        });
    }

    Result transferIn(const ArrayTensor<Device::CUDA, IT>& deviceInputBuffer) {
        BL_CHECK(this->copy(inputBuffer, deviceInputBuffer));
        return Result::SUCCESS;
    }

    Result transferResult() {
        BL_CHECK(this->copy(outputBuffer, batchStacker->getOutputBuffer()));
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

    using Detector = typename Modules::Detector<CF32, F32>;
    std::shared_ptr<Detector> detector;

    using Integrator = typename Modules::Integrator<F32, F32>;
    std::shared_ptr<Integrator> integrator;

    using OutputCaster = typename Modules::Caster<F32, OT>;
    std::shared_ptr<OutputCaster> outputCaster;

    using TimeStacker = typename Modules::Stacker<OT, OT>;
    std::shared_ptr<TimeStacker> timeStacker;

    using BatchStacker = typename Modules::Stacker<OT, OT>;
    std::shared_ptr<TimeStacker> batchStacker;

    Duet<ArrayTensor<Device::CUDA, IT>> inputBuffer;
    Duet<ArrayTensor<Device::CUDA, OT>> outputBuffer;

    PhasorTensor<Device::CUDA, CF32> inputPhasorBuffer;
};

struct FrbnnOp::Impl {
    // Derived configuration parameters.

    uint64_t numberOfBuffers;
    BlockShape inputShape;
    BlockShape outputShape;
    Map options;

    // State.

    Dispatcher dispatcher;
    OpFrbnnPipeline::Config config;
    std::shared_ptr<OpFrbnnPipeline> pipeline;

    // Metrics.

    std::thread metricsThread;
    bool metricsThreadRunning;
    void metricsLoop();
};

void FrbnnOp::initialize() {
    // Register custom types.
    register_converter<BlockShape>();
    register_converter<Map>();

    // Allocate memory.
    pimpl = new Impl();

    // Initialize operator.
    Operator::initialize();
}

FrbnnOp::~FrbnnOp() {
    delete pimpl;
}

void FrbnnOp::setup(OperatorSpec& spec) {
    spec.input<DspBlock>("dsp_block_in")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));
    spec.output<DspBlock>("dsp_block_out")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));

    spec.param(numberOfBuffers_, "number_of_buffers");
    spec.param(inputShape_, "input_shape");
    spec.param(outputShape_, "output_shape");
    spec.param(options_, "options");
}

void FrbnnOp::start() {
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

        .integratorSize = FetchMap<U64>(pimpl->options, "integrator_size", 32),
        .integratorRate = FetchMap<U64>(pimpl->options, "integrator_rate", 1),

        .timeStackerAxis = FetchMap<U64>(pimpl->options, "time_stacker_axis", 2),
        .timeStackerMultiplier = FetchMap<U64>(pimpl->options, "time_stacker_multiplier", 8),

        .batchStackerAxis = FetchMap<U64>(pimpl->options, "batch_stacker_axis", 0),
        .batchStackerMultiplier = FetchMap<U64>(pimpl->options, "batch_stacker_multiplier", 32),

        .inputCasterBlockSize = FetchMap<U64>(pimpl->options, "input_caster_block_size", 512),
        .integratorBlockSize = FetchMap<U64>(pimpl->options, "integrator_block_size", 512),
        .detectorBlockSize = FetchMap<U64>(pimpl->options, "detector_block_size", 512),
        .outputCasterBlockSize = FetchMap<U64>(pimpl->options, "output_caster_block_size", 512),
        .batchStackerBlockSize = FetchMap<U64>(pimpl->options, "batch_stacker_block_size", 512),
        .timeStackerBlockSize = FetchMap<U64>(pimpl->options, "time_stacker_block_size", 512),
    };
    pimpl->pipeline = std::make_shared<OpFrbnnPipeline>(pimpl->config);

    // Initialize Dispatcher.

    // TODO: Make number of buffers configurable.
    pimpl->dispatcher.template initialize<F32>(pimpl->numberOfBuffers, pimpl->config.outputShape);

    // Start metrics thread.

    pimpl->metricsThreadRunning = true;
    pimpl->metricsThread = std::thread([&]{
        pimpl->metricsLoop();
    });
}

void FrbnnOp::stop() {
    // Stop metrics thread.

    pimpl->metricsThreadRunning = false;
    if (pimpl->metricsThread.joinable()) {
        pimpl->metricsThread.join();
    }
}

void FrbnnOp::compute(InputContext& input, OutputContext& output, ExecutionContext&) {
    auto receiveCallback = [&](){
        return input.receive<DspBlock>("dsp_block_in").value();
    };

    auto convertInputCallback = [&](DspBlock& data){
        ArrayTensor<Device::CUDA, CF32> deviceInputBuffer(data.tensor->data(), pimpl->config.inputShape);
        return pimpl->pipeline->transferIn(deviceInputBuffer);
    };

    auto convertOutputCallback = [&](DspBlock& data){
        ArrayTensor<Device::CUDA, F32> deviceOutputBuffer(data.tensor->data(), pimpl->config.outputShape);
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

void FrbnnOp::Impl::metricsLoop() {
    while (metricsThreadRunning) {
        HOLOSCAN_LOG_INFO("Frbnn Operator:");
        dispatcher.metrics();

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

}  // namespace stelline::operators::blade
