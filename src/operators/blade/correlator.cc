#include <blade/base.hh>
#include <blade/bundles/generic/mode_x.hh>

#include <stelline/operators/blade/base.hh>
#include <stelline/types.hh>

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
    // Configuration parameters (derived).

    BlockShape inputShape;
    BlockShape outputShape;

    // State.

    Dispatcher dispatcher;
    std::shared_ptr<OpCorrelatorPipeline> pipeline;
    ArrayShape bladeInputShape;
    ArrayShape bladeOutputShape;

    // Metrics.

    std::thread metricsThread;
    bool metricsThreadRunning;
    void metricsLoop();
};

void CorrelatorOp::initialize() {
    // Register custom types.
    register_converter<BlockShape>();

    // Allocate memory.
    pimpl = new Impl();

    // Initialize operator.
    Operator::initialize();
}

CorrelatorOp::~CorrelatorOp() {
    delete pimpl;
}

void CorrelatorOp::setup(OperatorSpec& spec) {
    spec.input<DspBlock>("dsp_block_in")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));
    spec.output<DspBlock>("dsp_block_out")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));

    spec.param(inputShape_, "input_shape");
    spec.param(outputShape_, "output_shape");
}

void CorrelatorOp::start() {
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

    OpCorrelatorPipeline::Config config = {
        .inputShape = pimpl->bladeInputShape,
        .outputShape = pimpl->bladeOutputShape,

        .preCorrelatorStackerMultiplier = 1,

        .correlatorIntegrationRate = 8192,
        .correlatorConjugateAntennaIndex = 1,
    };
    pimpl->pipeline = std::make_shared<OpCorrelatorPipeline>(config);

    // Initialize Dispatcher.

    // TODO: Make number of buffers configurable.
    pimpl->dispatcher.template initialize<CF32>(4, pimpl->bladeOutputShape);

    // Start metrics thread.

    pimpl->metricsThreadRunning = true;
    pimpl->metricsThread = std::thread([&]{
        pimpl->metricsLoop();
    });
}

void CorrelatorOp::stop() {
    // Stop metrics thread.

    pimpl->metricsThreadRunning = false;
    if (pimpl->metricsThread.joinable()) {
        pimpl->metricsThread.join();
    }
}

void CorrelatorOp::compute(InputContext& input, OutputContext& output, ExecutionContext&) {
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

void CorrelatorOp::Impl::metricsLoop() {
    while (metricsThreadRunning) {
        HOLOSCAN_LOG_INFO("Correlator Operator:");
        dispatcher.metrics();

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

}  // namespace stelline::operators::blade
