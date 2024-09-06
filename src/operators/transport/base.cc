#include <stelline/operators/transport/base.hh>
#include <stelline/utils/juggler.hh>

#include <matx.h>
#include <holoscan/operators/advanced_network/adv_network_rx.h>

#include "types.hh"
#include "block.hh"

using namespace holoscan;
using namespace holoscan::ops;

namespace stelline::operators::transport {

struct ReceiverOp::Impl {
    // Configuration parameters (derived).

    BlockShape totalBlock;
    BlockShape partialBlock;
    BlockShape offsetBlock;
    uint64_t concurrentBlocks;

    // Cache parameters.

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

    Juggler<Tensor> blockTensorPool;

    // Metrics.

    std::thread metricsThread;
    bool metricsThreadRunning;
    void metricsLoop();

    uint64_t lostPackets;
    uint64_t receivedPackets;
    uint64_t evictedPackets;
    uint64_t receivedBlocks;
    uint64_t computedBlocks;
    uint64_t lostBlocks;
    std::set<uint64_t> allAntennas;
    std::set<uint64_t> allChannels;
    std::set<uint64_t> filteredAntennas;
    std::set<uint64_t> filteredChannels;
    std::chrono::microseconds avgBurstReleaseTime;

    // Burst collector.

    std::thread burstCollectorThread;
    bool burstCollectorThreadRunning;
    void burstCollectorLoop();

    std::mutex burstCollectorMutex;
    std::unordered_set<std::shared_ptr<AdvNetBurstParams>> bursts;
};

void ReceiverOp::initialize() {
    // Register custom types.
    register_converter<BlockShape>();

    // Allocate memory.
    pimpl = new Impl();

    // Initialize operator.
    Operator::initialize();
}

ReceiverOp::~ReceiverOp() {
    delete pimpl;
}

void ReceiverOp::setup(OperatorSpec& spec) {
    spec.input<AdvNetBurstParams>("burst_in");
    spec.output<std::shared_ptr<Tensor>>("dsp_block_out").connector(IOSpec::ConnectorType::kDoubleBuffer, holoscan::Arg("capacity", 32UL));

    spec.param(concurrentBlocks_, "concurrent_blocks");
    spec.param(totalBlock_, "total_block");
    spec.param(partialBlock_, "partial_block");
    spec.param(offsetBlock_, "offset_block");
}

void ReceiverOp::start() {
    // Convert Parameters to variables.

    pimpl->totalBlock = totalBlock_.get();
    pimpl->partialBlock = partialBlock_.get();
    pimpl->offsetBlock = offsetBlock_.get();
    pimpl->concurrentBlocks = concurrentBlocks_.get();

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

    pimpl->timestampCutoff = 0;
    pimpl->packetDuration = pimpl->partialBlock.numberOfSamples;
    pimpl->blockDuration = pimpl->slots.numberOfSamples * pimpl->packetDuration;

    // Allocate blocks.

    for (uint64_t i = 0; i < pimpl->concurrentBlocks; i++) {
        pimpl->idleQueue.push(std::make_shared<Block>(pimpl->packetsPerBlock));
    }

    // Start reporting thread.

    pimpl->metricsThreadRunning = true;
    pimpl->metricsThread = std::thread([&]{
        pimpl->metricsLoop();
    });

    // Start burst collector thread.

    pimpl->burstCollectorThreadRunning = true;
    pimpl->burstCollectorThread = std::thread([&]{
        pimpl->burstCollectorLoop();
    });

    // Print static variables.

    HOLOSCAN_LOG_INFO("Packets per block: {}", pimpl->packetsPerBlock);

    // Allocate block tensor pool.

    pimpl->blockTensorPool.resize(512, [&]{
        auto tensor = matx::make_tensor<std::complex<float>>({
            static_cast<int64_t>(pimpl->totalBlock.numberOfAntennas),
            static_cast<int64_t>(pimpl->totalBlock.numberOfChannels),
            static_cast<int64_t>(pimpl->totalBlock.numberOfSamples),
            static_cast<int64_t>(pimpl->totalBlock.numberOfPolarizations)
        });
        return std::make_shared<Tensor>(tensor.GetDLPackTensor());
    });
}

void ReceiverOp::stop() {
    pimpl->idleQueue = {};
    pimpl->receiveQueue = {};
    pimpl->computeQueue = {};

    pimpl->blockMap.clear();

    // Stop burst collector thread.

    pimpl->burstCollectorThreadRunning = false;
    if (pimpl->burstCollectorThread.joinable()) {
        pimpl->burstCollectorThread.join();
    }

    // Stop metrics thread.

    pimpl->metricsThreadRunning = false;
    if (pimpl->metricsThread.joinable()) {
        pimpl->metricsThread.join();
    }
}

void ReceiverOp::compute(InputContext& input, OutputContext& output, ExecutionContext&) {
    auto burst = input.receive<std::shared_ptr<AdvNetBurstParams>>("burst_in").value();

    if (!burst) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(pimpl->burstCollectorMutex);
        pimpl->bursts.insert(burst);
    }

    for (int64_t i = 0; i < adv_net_get_num_pkts(burst); i++) {
        const auto& burstPacketIndex = i;

        const auto* p = reinterpret_cast<uint8_t*>(adv_net_get_cpu_pkt_ptr(burst, burstPacketIndex));
        const VoltagePacket packet(p + TransportHeaderSize);

        uint64_t antennaIndex = packet.antennaId;
        uint64_t channelIndex = packet.channelNumber;

        pimpl->allAntennas.insert(antennaIndex);
        pimpl->allChannels.insert(channelIndex);

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

        if (packet.timestamp < pimpl->timestampCutoff) {
            pimpl->evictedPackets += 1;
            continue;
        }

        uint64_t blockTimeIndex = packet.timestamp / pimpl->blockDuration;
        uint64_t blockPacketTimeIndex = (packet.timestamp % pimpl->blockDuration) / pimpl->partialBlock.numberOfSamples;

        if (!pimpl->blockMap.contains(blockTimeIndex)) {
            if (pimpl->idleQueue.empty()) {
                if (pimpl->receiveQueue.empty()) {
                    pimpl->lostPackets += 1;
                    continue;
                }

                auto block = pimpl->receiveQueue.front();
                pimpl->receiveQueue.pop();
                pimpl->blockMap.erase(block->timeIndex());
                block->destroy();
                pimpl->idleQueue.push(block);
                pimpl->lostBlocks += 1;
            }

            auto block = pimpl->idleQueue.front();
            pimpl->idleQueue.pop();
            block->create(blockTimeIndex);
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

    // Scan receive queue for complete blocks.

    while (!pimpl->receiveQueue.empty()) {
        auto block = pimpl->receiveQueue.front();
        pimpl->receiveQueue.pop();

        if (block->isComplete()) {
            pimpl->blockMap.erase(block->timeIndex());
            std::shared_ptr<Tensor> tensor;
            while ((tensor = pimpl->blockTensorPool.get()) == nullptr) {
                throw std::runtime_error("Failed to allocate tensor from pool.");
            }
            block->compute(tensor, pimpl->totalBlock, pimpl->partialBlock, pimpl->slots);
            pimpl->computeQueue.push(block);
            pimpl->receivedBlocks += 1;
        } else {
            pimpl->swapQueue.push(block);
        }
    }

    while (!pimpl->swapQueue.empty()) {
        pimpl->receiveQueue.push(pimpl->swapQueue.front());
        pimpl->swapQueue.pop();
    }

    // Scan compute queue for blocks ready to be released.

    while (!pimpl->computeQueue.empty()) {
        auto block = pimpl->computeQueue.front();
        pimpl->computeQueue.pop();

        if (!block->isProcessing()) {
            output.emit(block->outputTensor(), "dsp_block_out");
            block->destroy();
            pimpl->idleQueue.push(block);
            pimpl->computedBlocks += 1;
        } else {
            pimpl->swapQueue.push(block);
        }
    }

    while (!pimpl->swapQueue.empty()) {
        pimpl->computeQueue.push(pimpl->swapQueue.front());
        pimpl->swapQueue.pop();
    }

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

void ReceiverOp::Impl::burstCollectorLoop() {
    std::chrono::microseconds totalRuntime(0);
    uint64_t numIterations = 0;

    while (burstCollectorThreadRunning) {
        auto startTime = std::chrono::steady_clock::now();

        const auto numReleasedBursts = [&]{
            std::unordered_set<std::shared_ptr<AdvNetBurstParams>> stale;

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
                adv_net_free_all_burst_pkts_and_burst(burst);
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

void ReceiverOp::Impl::metricsLoop() {
    while (metricsThreadRunning) {
        HOLOSCAN_LOG_INFO("Blocks    : {} received, {} computed, {} lost", receivedBlocks, computedBlocks, lostBlocks);
        HOLOSCAN_LOG_INFO("Packets   : {} evicted, {} received, {} lost", evictedPackets, receivedPackets, lostPackets);
        HOLOSCAN_LOG_INFO("In-Flight : {} idle, {} receive, {} compute", idleQueue.size(), receiveQueue.size(), computeQueue.size());
        HOLOSCAN_LOG_INFO("Bursts    : {} in-flight, {} us average per-burst release time", bursts.size(), avgBurstReleaseTime.count());
        HOLOSCAN_LOG_INFO("Mem Pool  : {} available, {} referenced", blockTensorPool.available(), blockTensorPool.referenced());
        HOLOSCAN_LOG_INFO("Fine Packet Count:");
        HOLOSCAN_LOG_INFO("   All antennas     : {}", allAntennas);
        HOLOSCAN_LOG_INFO("   Filtered antennas: {}", filteredAntennas);
        HOLOSCAN_LOG_INFO("   All channels     : {}", allChannels);
        HOLOSCAN_LOG_INFO("   Filtered channels: {}", filteredChannels);

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

}  // namespace stelline::operators::transport
