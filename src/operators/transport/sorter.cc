#include <stelline/types.hh>
#include <stelline/operators/transport/base.hh>
#include <fmt/format.h>

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
}

void SorterOp::stop() {
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

stelline::StoreInterface::MetricsMap SorterOp::collectMetricsMap() {
    stelline::StoreInterface::MetricsMap metrics;
    metrics["rejected_blocks"] = fmt::format("{}", pimpl->numberOfRejectedBlocks);
    metrics["pool_size"] = fmt::format("{}", pimpl->pool.size());
    return metrics;
}

std::string SorterOp::collectMetricsString() {
    const auto metrics = collectMetricsMap();
    return fmt::format("Sorter Operator:\n"
                       "  Number of Rejected Blocks: {}\n"
                       "  Pool Size: {}",
                       metrics.at("rejected_blocks"),
                       metrics.at("pool_size"));
}

}  // namespace stelline::operators::transport
