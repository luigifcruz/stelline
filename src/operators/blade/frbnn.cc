#include <blade/base.hh>
#include <blade/modules/beamformer/ata.hh>
#include <blade/modules/integrator.hh>
#include <blade/modules/stacker.hh>
#include <blade/modules/caster.hh>
#include <blade/modules/detector.hh>

#include <stelline/operators/blade/base.hh>
#include <stelline/types.hh>

#include "dispatcher.hh"

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

        U64 integratorSize = 32;
        U64 integratorRate = 1;

        U64 timeStackerAxis = 2;
        U64 timeStackerMultiplier = 8;

        U64 batchStackerAxis = 0;
        U64 batchStackerMultiplier = 32;

        U64 inputCasterBlockSize = 512;
        U64 integratorBlockSize = 512;
        U64 detectorBlockSize = 512;
        U64 outputCasterBlockSize = 512;
        U64 batchStackerBlockSize = 512;
        U64 timeStackerBlockSize = 512;
    };

    explicit OpFrbnnPipeline(const Config& config)
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

        this->connect(integrator, {
            .size = config.integratorSize,
            .rate = config.integratorRate,

            .blockSize = config.integratorBlockSize,
        }, {
            .buf = beamformer->getOutputBuffer(),
        });

        this->connect(detector, {
            .integrationRate = 1,
            .numberOfOutputPolarizations = 1,

            .blockSize = config.detectorBlockSize,
        }, {
            .buf = integrator->getOutputBuffer(),
        });

        this->connect(outputCaster, {
            .blockSize = config.outputCasterBlockSize,
        }, {
            .buf = detector->getOutputBuffer(),
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

    using Integrator = typename Modules::Integrator<CF32, CF32>;
    std::shared_ptr<Integrator> integrator;

    using Detector = typename Modules::Detector<CF32, F32>;
    std::shared_ptr<Detector> detector;

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
    // Configuration parameters (derived).

    BlockShape inputShape;
    BlockShape outputShape;

    // State.

    Dispatcher dispatcher;
    std::shared_ptr<OpFrbnnPipeline> pipeline;
    ArrayShape bladeInputShape;
    ArrayShape bladeOutputShape;
};

void FrbnnOp::initialize() {
    // Register custom types.
    register_converter<BlockShape>();

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

    spec.param(inputShape_, "input_shape");
    spec.param(outputShape_, "output_shape");
}

void FrbnnOp::start() {
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

    OpFrbnnPipeline::Config config = {
        .inputShape = pimpl->bladeInputShape,
        .outputShape = pimpl->bladeOutputShape
    };
    pimpl->pipeline = std::make_shared<OpFrbnnPipeline>(config);

    // Initialize Dispatcher.

    // TODO: Make number of buffers configurable.
    pimpl->dispatcher.template initialize<F32>(4, pimpl->bladeOutputShape);
}

void FrbnnOp::compute(InputContext& input, OutputContext& output, ExecutionContext&) {
    auto receiveCallback = [&](){
        return input.receive<DspBlock>("dsp_block_in").value();
    };

    auto convertInputCallback = [&](Dispatcher::Job& job){
        ArrayTensor<Device::CUDA, CF32> deviceInputBuffer(job.input.tensor->data(), pimpl->bladeInputShape);
        return pimpl->pipeline->transferIn(deviceInputBuffer);
    };

    auto convertOutputCallback = [&](Dispatcher::Job& job){
        ArrayTensor<Device::CUDA, F32> deviceOutputBuffer(job.output.tensor->data(), pimpl->bladeOutputShape);
        return pimpl->pipeline->transferOut(deviceOutputBuffer);
    };

    auto emitCallback = [&](Dispatcher::Job& job){
        output.emit(job.output, "dsp_block_out");
    };

    pimpl->dispatcher.run(pimpl->pipeline, receiveCallback, convertInputCallback, convertOutputCallback, emitCallback);
}

}  // namespace stelline::operators::blade
