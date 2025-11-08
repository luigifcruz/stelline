#include <stelline/types.hh>
#include <stelline/operators/transport/base.hh>
#include <stelline/utils/juggler.hh>

#include <matx.h>

#include "types.hh"
#include "block.hh"

using namespace holoscan;
using namespace holoscan::ops;

namespace stelline::operators::transport {

struct DummyReceiverOp::Impl {
    // Configuration parameters (derived).

    BlockShape totalBlock;

    // Cache parameters.

    uint64_t timestamp;

    // State.

    std::shared_ptr<holoscan::Tensor> tensor;

    // Memory pools.


    // Release helpers.

    void releaseReceivedBlocks();
    void releaseComputedBlocks(OutputContext& output);

    // Burst collector.

    std::thread burstCollectorThread;
    bool burstCollectorThreadRunning;
    void burstCollectorLoop();

    std::mutex burstCollectorMutex;
};

void DummyReceiverOp::initialize() {
    // Register custom types.
    register_converter<BlockShape>();

    // Allocate memory.
    pimpl = new Impl();

    // Initialize operator.
    Operator::initialize();
}

DummyReceiverOp::~DummyReceiverOp() {
    delete pimpl;
}

void DummyReceiverOp::setup(OperatorSpec& spec) {
    spec.output<DspBlock>("dsp_block_out")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));

    spec.param(totalBlock_, "total_block");
}

void DummyReceiverOp::start() {
    // Convert Parameters to variables.

    pimpl->totalBlock = totalBlock_.get();

    // Validate configuration.

    assert(pimpl->totalBlock.numberOfAntennas != 0);
    assert(pimpl->totalBlock.numberOfChannels != 0);
    assert(pimpl->totalBlock.numberOfSamples != 0);
    assert(pimpl->totalBlock.numberOfPolarizations != 0);

    pimpl->timestamp = 0;

    // Allocate tensor.

    pimpl->tensor = std::make_shared<Tensor>(matx::make_tensor<cuda::std::complex<float>>({
        static_cast<int64_t>(pimpl->totalBlock.numberOfAntennas),
        static_cast<int64_t>(pimpl->totalBlock.numberOfChannels),
        static_cast<int64_t>(pimpl->totalBlock.numberOfSamples),
        static_cast<int64_t>(pimpl->totalBlock.numberOfPolarizations)
    }, matx::MATX_DEVICE_MEMORY).ToDlPack());
}

void DummyReceiverOp::stop() {}

void DummyReceiverOp::compute(InputContext& input, OutputContext& output, ExecutionContext&) {
    HOLOSCAN_LOG_INFO("Faking block {}.", pimpl->timestamp);
    DspBlock outputBlock = {
        .timestamp = pimpl->timestamp,
        .tensor = pimpl->tensor,
    };
    output.emit(outputBlock, "dsp_block_out");
    pimpl->timestamp += 1;

    // Check for execution errors.

    cudaError_t val;
    if ((val = cudaPeekAtLastError()) != cudaSuccess) {
        // Get error message.
        const char* err = cudaGetErrorString(val);

        // Print error message.
        HOLOSCAN_LOG_ERROR("CUDA Error: {}", err);

        // Wait for metrics thread to print latest statistics.
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // Throw exception.
        throw std::runtime_error(err);
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));
    HOLOSCAN_LOG_WARN("Finished sleeping...");
}

}  // namespace stelline::operators::transport
