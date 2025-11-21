#include <stelline/types.hh>
#include <stelline/operators/transport/base.hh>

#include "types.hh"
#include "block.hh"

using namespace holoscan;
using namespace holoscan::ops;

namespace stelline::operators::transport {

struct SorterOp::Impl {
    // State.

    uint64_t depth = 0;
    uint64_t timestamp = 0;
    std::unordered_map<uint64_t, std::shared_ptr<holoscan::Tensor>> pool;

    // Metrics.

    std::thread metricsThread;
    bool metricsThreadRunning;
    void metricsLoop();

    uint64_t numberOfRejectedBlocks = 0;
};

void SorterOp::initialize() {
    // Allocate memory.
    pimpl = new Impl();

    // Initialize operator.
    Operator::initialize();
}

SorterOp::~SorterOp() {
    delete pimpl;
}

void SorterOp::setup(OperatorSpec& spec) {
    spec.input<std::shared_ptr<holoscan::Tensor>>("dsp_block_in")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                    holoscan::Arg("capacity", 1024UL));
    spec.output<std::shared_ptr<holoscan::Tensor>>("dsp_block_out")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                    holoscan::Arg("capacity", 1024UL));
    spec.param(depth_, "depth");
}

void SorterOp::start() {
    // Convert Parameters to variables.

    pimpl->depth = depth_.get();

    // Check parameters.

    if (pimpl->depth == 0) {
        HOLOSCAN_LOG_ERROR("Sorter depth must be greater than 0.");
        throw std::runtime_error("Parameter error.");
    }

    // Start reporting thread.

    pimpl->metricsThreadRunning = true;
    pimpl->metricsThread = std::thread([&]{
        pimpl->metricsLoop();
    });
}

void SorterOp::stop() {
    // Stop metrics thread.

    pimpl->metricsThreadRunning = false;
    if (pimpl->metricsThread.joinable()) {
        pimpl->metricsThread.join();
    }
}

void SorterOp::compute(InputContext& input, OutputContext& output, ExecutionContext&) {
    const auto& meta = metadata();

    // Receive all blocks.

    while (auto ptr = input.receive<std::shared_ptr<holoscan::Tensor>>("dsp_block_in")) {
        auto tensor = ptr.value();

        // Get the timestamp from the metadata.

        const auto& timestamp = meta->get<uint64_t>("timestamp");

        // Check incoming block timestamp.

        if (pimpl->timestamp > timestamp) {
            HOLOSCAN_LOG_WARN("Monotonic timestamps required. Discarding block.");
            pimpl->numberOfRejectedBlocks++;
            continue;
        }

        // Add block to pool.

        pimpl->pool[timestamp] = tensor;
    }

    // Sort blocks by timestamp.

    while (pimpl->pool.size() > pimpl->depth) {
        uint64_t minTimestamp = std::numeric_limits<uint64_t>::max();
        for (auto& [timestamp, _] : pimpl->pool) {
            if (timestamp < minTimestamp) {
                minTimestamp = timestamp;
            }
        }

        meta->set("timestamp", minTimestamp);
        output.emit(pimpl->pool.at(minTimestamp), "dsp_block_out");

        pimpl->pool.erase(minTimestamp);
        pimpl->timestamp = minTimestamp;
    }
}

void SorterOp::Impl::metricsLoop() {
    while (metricsThreadRunning) {
        HOLOSCAN_LOG_INFO("Sorter Operator:");
        HOLOSCAN_LOG_INFO("  Number of Rejected Blocks: {}", numberOfRejectedBlocks);

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

}  // namespace stelline::operators::transport
