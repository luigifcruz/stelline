#include <stelline/types.hh>
#include <stelline/operators/transport/base.hh>
#include <stelline/utils/juggler.hh>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <vector>

#include <matx.h>
#include <advanced_network/common.h>

#include "types.hh"
#include "block.hh"

#define RX_HEADER 0
#define RX_DATA 1

using namespace holoscan;
using namespace holoscan::ops;
using namespace holoscan::advanced_network;

namespace stelline::operators::transport {

struct AtaReceiverOp::Impl {
    // Configuration parameters (derived).

    BlockShape totalBlock;
    BlockShape partialBlock;
    BlockShape offsetBlock;
    uint64_t maxConcurrentBlocks;
    uint64_t outputPoolSize;
    uint64_t packetHeaderSize;
    uint64_t packetHeaderOffset;
    bool enableCsvLogging;

    // Cache parameters.

    uint64_t latestBlockTimeIndex;
    uint64_t latestTimestamp;
    uint64_t timestampCutoff;
    uint64_t blockDuration;
    uint64_t packetDuration;
    uint64_t packetsPerBlock;
    BlockShape slots;

    // State.

    std::queue<std::shared_ptr<Block>> idleQueue;
    std::queue<std::shared_ptr<Block>> receiveQueue;
    std::queue<std::shared_ptr<Block>> computeQueue;
    std::queue<std::shared_ptr<Block>> swapQueue;

    std::map<uint64_t, std::shared_ptr<Block>> blockMap;

    // Memory pools.

    Juggler<holoscan::Tensor> blockTensorPool;

    // Metrics.

    uint64_t lostPackets;
    uint64_t receivedPackets;
    uint64_t evictedPackets;
    uint64_t receivedBlocks;
    uint64_t computedBlocks;
    uint64_t lostBlocks;
    std::set<uint64_t> allAntennas;
    std::set<uint64_t> allChannels;
    std::set<uint64_t> payloadSizes;
    std::set<uint64_t> filteredAntennas;
    std::set<uint64_t> filteredChannels;
    std::chrono::microseconds avgBurstReleaseTime;

    // Release helpers.

    void releaseReceivedBlocks();
    void releaseComputedBlocks(const std::shared_ptr<MetadataDictionary>& meta, OutputContext& output);

    // Burst collector.

    std::thread burstCollectorThread;
    bool burstCollectorThreadRunning;
    void burstCollectorLoop();

    std::mutex burstCollectorMutex;
    std::unordered_set<std::shared_ptr<BurstParams>> bursts;
};

void AtaReceiverOp::initialize() {
    // Register custom types.
    register_converter<BlockShape>();

    // Allocate memory.
    pimpl = new Impl();

    // Initialize operator.
    Operator::initialize();
}

AtaReceiverOp::~AtaReceiverOp() {
    delete pimpl;
}

void AtaReceiverOp::setup(OperatorSpec& spec) {
    spec.output<std::shared_ptr<holoscan::Tensor>>("dsp_block_out")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));

    spec.param(maxConcurrentBlocks_, "max_concurrent_blocks");
    spec.param(packetHeaderSize_, "packet_header_size");
    spec.param(packetHeaderOffset_, "packet_header_offset");
    spec.param(totalBlock_, "total_block");
    spec.param(partialBlock_, "partial_block");
    spec.param(offsetBlock_, "offset_block");
    spec.param(outputPoolSize_, "output_pool_size");
    spec.param(enableCsvLogging_, "enable_csv_logging");
}

void AtaReceiverOp::start() {
    // Convert Parameters to variables.

    pimpl->totalBlock = totalBlock_.get();
    pimpl->packetHeaderSize = packetHeaderSize_.get();
    pimpl->packetHeaderOffset = packetHeaderOffset_.get();
    pimpl->partialBlock = partialBlock_.get();
    pimpl->offsetBlock = offsetBlock_.get();
    pimpl->maxConcurrentBlocks = maxConcurrentBlocks_.get();
    pimpl->outputPoolSize = outputPoolSize_.get();
    pimpl->enableCsvLogging = enableCsvLogging_.get();

    // Validate configuration.

    assert((pimpl->offsetBlock.numberOfAntennas % pimpl->partialBlock.numberOfAntennas) == 0);
    assert(pimpl->offsetBlock.numberOfSamples == 0);
    assert(pimpl->offsetBlock.numberOfPolarizations == 0);

    assert(pimpl->totalBlock.numberOfAntennas != 0);
    assert(pimpl->totalBlock.numberOfChannels != 0);
    assert(pimpl->totalBlock.numberOfSamples != 0);
    assert(pimpl->totalBlock.numberOfPolarizations != 0);

    assert(pimpl->partialBlock.numberOfAntennas != 0);
    assert(pimpl->partialBlock.numberOfChannels != 0);
    assert(pimpl->partialBlock.numberOfSamples != 0);
    assert(pimpl->partialBlock.numberOfPolarizations != 0);

    assert((pimpl->totalBlock.numberOfAntennas % pimpl->partialBlock.numberOfAntennas) == 0);
    assert((pimpl->totalBlock.numberOfChannels % pimpl->partialBlock.numberOfChannels) == 0);
    assert((pimpl->totalBlock.numberOfSamples % pimpl->partialBlock.numberOfSamples) == 0);
    assert((pimpl->totalBlock.numberOfPolarizations % pimpl->partialBlock.numberOfPolarizations) == 0);

    // Calculate total number of packets per block.

    pimpl->packetsPerBlock = 1;
    pimpl->packetsPerBlock *= pimpl->slots.numberOfAntennas = pimpl->totalBlock.numberOfAntennas / pimpl->partialBlock.numberOfAntennas;
    pimpl->packetsPerBlock *= pimpl->slots.numberOfChannels = pimpl->totalBlock.numberOfChannels / pimpl->partialBlock.numberOfChannels;
    pimpl->packetsPerBlock *= pimpl->slots.numberOfSamples = pimpl->totalBlock.numberOfSamples / pimpl->partialBlock.numberOfSamples;
    pimpl->packetsPerBlock *= pimpl->slots.numberOfPolarizations = pimpl->totalBlock.numberOfPolarizations / pimpl->partialBlock.numberOfPolarizations;

    // Calculate the block duration.

    pimpl->latestTimestamp = 0;
    pimpl->timestampCutoff = 0;
    pimpl->packetDuration = pimpl->partialBlock.numberOfSamples;
    pimpl->blockDuration = pimpl->slots.numberOfSamples * pimpl->packetDuration;

    pimpl->lostPackets = 0;
    pimpl->receivedPackets = 0;
    pimpl->evictedPackets = 0;
    pimpl->receivedBlocks = 0;
    pimpl->computedBlocks = 0;
    pimpl->lostBlocks = 0;
    pimpl->allAntennas.clear();
    pimpl->allChannels.clear();
    pimpl->payloadSizes.clear();
    pimpl->filteredAntennas.clear();
    pimpl->filteredChannels.clear();
    pimpl->avgBurstReleaseTime = std::chrono::microseconds(0);

    // Allocate blocks.

    for (uint64_t i = 0; i < pimpl->maxConcurrentBlocks; i++) {
        pimpl->idleQueue.push(std::make_shared<Block>(pimpl->packetsPerBlock));
    }

    // Start burst collector thread.

    pimpl->burstCollectorThreadRunning = true;
    pimpl->burstCollectorThread = std::thread([&]{
        pimpl->burstCollectorLoop();
    });

    // Allocate block tensor pool.

    pimpl->blockTensorPool.resize(pimpl->outputPoolSize, [&]{
        auto tensor = matx::make_tensor<cuda::std::complex<float>>({
            static_cast<int64_t>(pimpl->totalBlock.numberOfAntennas),
            static_cast<int64_t>(pimpl->totalBlock.numberOfChannels),
            static_cast<int64_t>(pimpl->totalBlock.numberOfSamples),
            static_cast<int64_t>(pimpl->totalBlock.numberOfPolarizations)
        }, matx::MATX_DEVICE_MEMORY);
        return std::make_shared<holoscan::Tensor>(tensor.ToDlPack());
    });
}

void AtaReceiverOp::stop() {
    pimpl->idleQueue = {};
    pimpl->receiveQueue = {};
    pimpl->computeQueue = {};

    pimpl->blockMap.clear();

    // Stop burst collector thread.

    pimpl->burstCollectorThreadRunning = false;
    if (pimpl->burstCollectorThread.joinable()) {
        pimpl->burstCollectorThread.join();
    }

    advanced_network::shutdown();
}

void AtaReceiverOp::compute(InputContext& input, OutputContext& output, ExecutionContext&) {
    BurstParams* burstPtr;
    if (get_rx_burst(&burstPtr) != Status::SUCCESS) {
        return;
    }
    auto burst = std::shared_ptr<BurstParams>(burstPtr, [](BurstParams*){});

    {
        std::lock_guard<std::mutex> lock(pimpl->burstCollectorMutex);
        pimpl->bursts.insert(burst);
    }

    for (int64_t i = 0; i < get_num_packets(burst.get()); i++) {
        const auto& burstPacketIndex = i;

        const auto* p = reinterpret_cast<uint8_t*>(get_segment_packet_ptr(burst.get(), RX_HEADER, burstPacketIndex));
        const VoltagePacket packet(p + pimpl->packetHeaderOffset);

        uint64_t antennaIndex = packet.antennaId;
        uint64_t channelIndex = packet.channelNumber;

        pimpl->allAntennas.insert(antennaIndex);
        pimpl->allChannels.insert(channelIndex);
        pimpl->payloadSizes.insert(get_segment_packet_length(burst.get(), RX_DATA, burstPacketIndex));

        antennaIndex -= pimpl->offsetBlock.numberOfAntennas;
        channelIndex -= pimpl->offsetBlock.numberOfChannels;

        if (antennaIndex >= pimpl->totalBlock.numberOfAntennas) {
            pimpl->evictedPackets += 1;
            continue;
        }

        if (channelIndex >= pimpl->totalBlock.numberOfChannels) {
            pimpl->evictedPackets += 1;
            continue;
        }

        pimpl->filteredAntennas.insert(antennaIndex + pimpl->offsetBlock.numberOfAntennas);
        pimpl->filteredChannels.insert(channelIndex + pimpl->offsetBlock.numberOfChannels);

        pimpl->latestTimestamp = std::max(pimpl->latestTimestamp, packet.timestamp);
        if (packet.timestamp < pimpl->timestampCutoff) {
            pimpl->evictedPackets += 1;
            continue;
        }

        uint64_t blockTimeIndex = packet.timestamp / pimpl->blockDuration;
        uint64_t blockPacketTimeIndex = (packet.timestamp % pimpl->blockDuration) / pimpl->partialBlock.numberOfSamples;

        pimpl->latestBlockTimeIndex = std::max(pimpl->latestBlockTimeIndex, blockTimeIndex);

        if (!pimpl->blockMap.contains(blockTimeIndex)) {
            if (pimpl->idleQueue.empty()) {
                if (pimpl->receiveQueue.empty()) {
                    pimpl->lostPackets += 1;
                    continue;
                }

                auto block = pimpl->receiveQueue.front();
                pimpl->receiveQueue.pop();
                pimpl->blockMap.erase(block->index());
                block->destroy();
                pimpl->idleQueue.push(block);
                pimpl->lostBlocks += 1;
            }

            auto block = pimpl->idleQueue.front();
            pimpl->idleQueue.pop();
            block->create(blockTimeIndex, packet.timestamp);
            pimpl->receiveQueue.push(block);
            pimpl->blockMap[blockTimeIndex] = block;
        }

        antennaIndex /= pimpl->partialBlock.numberOfAntennas;
        channelIndex /= pimpl->partialBlock.numberOfChannels;

        uint64_t blockPacketIndex = 0;
        blockPacketIndex += antennaIndex * pimpl->slots.numberOfChannels * pimpl->slots.numberOfSamples;
        blockPacketIndex += channelIndex * pimpl->slots.numberOfSamples;
        blockPacketIndex += blockPacketTimeIndex;

        auto& block = pimpl->blockMap[blockTimeIndex];
        block->addPacket(blockPacketIndex, burstPacketIndex, burst);

        pimpl->receivedPackets += 1;
    }

    // Run release helpers.

    pimpl->releaseReceivedBlocks();
    pimpl->releaseComputedBlocks(metadata(), output);

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
}

void AtaReceiverOp::Impl::releaseReceivedBlocks() {
    while (!receiveQueue.empty()) {
        auto block = receiveQueue.front();
        receiveQueue.pop();

        bool isStale = (latestBlockTimeIndex > block->index() + maxConcurrentBlocks);

        if (block->isComplete()) {
            blockMap.erase(block->index());
            std::shared_ptr<holoscan::Tensor> tensor;
            while ((tensor = blockTensorPool.get()) == nullptr) {
                HOLOSCAN_LOG_ERROR("Failed to allocate tensor from pool.");
                throw std::runtime_error("Failed to allocate tensor from pool.");
            }
            block->compute(tensor, totalBlock, partialBlock, slots);
            computeQueue.push(block);
            receivedBlocks += 1;
        } else if (isStale) {
            blockMap.erase(block->index());
            block->destroy();
            idleQueue.push(block);
            lostBlocks += 1;
        } else {
            swapQueue.push(block);
        }
    }

    while (!swapQueue.empty()) {
        receiveQueue.push(swapQueue.front());
        swapQueue.pop();
    }
}

void AtaReceiverOp::Impl::releaseComputedBlocks(const std::shared_ptr<MetadataDictionary>& meta, OutputContext& output) {
    while (!computeQueue.empty()) {
        auto block = computeQueue.front();
        computeQueue.pop();

        if (!block->isProcessing()) {
            meta->set("timestamp", block->timestamp());
            output.emit(block->outputTensor(), "dsp_block_out");
            block->destroy();
            idleQueue.push(block);
            computedBlocks += 1;
        } else {
            swapQueue.push(block);
        }
    }

    while (!swapQueue.empty()) {
        computeQueue.push(swapQueue.front());
        swapQueue.pop();
    }
}

void AtaReceiverOp::Impl::burstCollectorLoop() {
    std::chrono::microseconds totalRuntime(0);
    uint64_t numIterations = 1;

    while (burstCollectorThreadRunning) {
        auto startTime = std::chrono::steady_clock::now();

        const auto numReleasedBursts = [&]{
            std::unordered_set<std::shared_ptr<BurstParams>> stale;

            {
                std::lock_guard<std::mutex> lock(burstCollectorMutex);
                for (const auto& burst : bursts) {
                    if (burst.unique()) {
                        stale.insert(burst);
                    }
                }

                for (const auto& burst : stale) {
                    bursts.erase(burst);
                }
            }

            for (const auto& burst : stale) {
                free_all_packets_and_burst_rx(burst.get());
            }

            return stale.size();
        }();

        auto endTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

        totalRuntime += elapsedTime;
        numIterations += numReleasedBursts;
        avgBurstReleaseTime = totalRuntime / numIterations;

        std::chrono::microseconds sleepTime(10000 - elapsedTime.count());
        if (sleepTime.count() > 0) {
            std::this_thread::sleep_for(sleepTime);
        }
    }
}

stelline::StoreInterface::MetricsMap AtaReceiverOp::collectMetricsMap() {
    std::vector<uint64_t> blockTimes;
    blockTimes.reserve(pimpl->blockMap.size());
    for (const auto& [time, _] : pimpl->blockMap) {
        blockTimes.push_back(time);
    }

    stelline::StoreInterface::MetricsMap metrics;
    metrics["blocks_received"] = fmt::format("{}", pimpl->receivedBlocks);
    metrics["blocks_computed"] = fmt::format("{}", pimpl->computedBlocks);
    metrics["blocks_lost"] = fmt::format("{}", pimpl->lostBlocks);
    metrics["packets_evicted"] = fmt::format("{}", pimpl->evictedPackets);
    metrics["packets_received"] = fmt::format("{}", pimpl->receivedPackets);
    metrics["packets_lost"] = fmt::format("{}", pimpl->lostPackets);
    metrics["idle_queue"] = fmt::format("{}", pimpl->idleQueue.size());
    metrics["receive_queue"] = fmt::format("{}", pimpl->receiveQueue.size());
    metrics["compute_queue"] = fmt::format("{}", pimpl->computeQueue.size());
    metrics["bursts_in_flight"] = fmt::format("{}", pimpl->bursts.size());
    metrics["avg_burst_release_time_us"] = fmt::format("{}", pimpl->avgBurstReleaseTime.count());
    metrics["mem_pool_available"] = fmt::format("{}", pimpl->blockTensorPool.available());
    metrics["mem_pool_referenced"] = fmt::format("{}", pimpl->blockTensorPool.referenced());
    metrics["block_map_latest_time_index"] = fmt::format("{}", pimpl->latestBlockTimeIndex);
    metrics["block_map_usage"] = fmt::format("{}/{}", pimpl->blockMap.size(), pimpl->maxConcurrentBlocks);
    metrics["block_map_times"] = fmt::format("[{}]", fmt::join(blockTimes, ","));
    metrics["payload_sizes"] = fmt::format("[{}]", fmt::join(pimpl->payloadSizes, ","));
    metrics["all_antennas"] = fmt::format("[{}]", fmt::join(pimpl->allAntennas, ","));
    metrics["filtered_antennas"] = fmt::format("[{}]", fmt::join(pimpl->filteredAntennas, ","));
    metrics["all_channels"] = fmt::format("[{}]", fmt::join(pimpl->allChannels, ","));
    metrics["filtered_channels"] = fmt::format("[{}]", fmt::join(pimpl->filteredChannels, ","));
    metrics["latest_timestamp"] = fmt::format("{}", pimpl->latestTimestamp);

    return metrics;
}

std::string AtaReceiverOp::collectMetricsString() {
    const auto metrics = collectMetricsMap();
    return fmt::format(
        "Transport Operator:\n"
        "  Blocks    : {} received, {} computed, {} lost\n"
        "  Packets   : {} evicted, {} received, {} lost\n"
        "  In-Flight : {} idle, {} receive, {} compute\n"
        "  Bursts    : {} in-flight, {} us average per-burst release time\n"
        "  Mem Pool  : {} available, {} referenced\n"
        "  Block Map : latest time index {}, usage {}, all block times {}\n"
        "  Fine Packet Count:\n"
        "    Payload Sizes    : {}\n"
        "    All antennas     : {}\n"
        "    Filtered antennas: {}\n"
        "    All channels     : {}\n"
        "    Filtered channels: {}",
        metrics.at("blocks_received"),
        metrics.at("blocks_computed"),
        metrics.at("blocks_lost"),
        metrics.at("packets_evicted"),
        metrics.at("packets_received"),
        metrics.at("packets_lost"),
        metrics.at("idle_queue"),
        metrics.at("receive_queue"),
        metrics.at("compute_queue"),
        metrics.at("bursts_in_flight"),
        metrics.at("avg_burst_release_time_us"),
        metrics.at("mem_pool_available"),
        metrics.at("mem_pool_referenced"),
        metrics.at("block_map_latest_time_index"),
        metrics.at("block_map_usage"),
        metrics.at("block_map_times"),
        metrics.at("payload_sizes"),
        metrics.at("all_antennas"),
        metrics.at("filtered_antennas"),
        metrics.at("all_channels"),
        metrics.at("filtered_channels"));
}

}  // namespace stelline::operators::transport
