#include <chrono>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "jetstream/backend/devices/cuda/helpers.hh"

#include <daqiri/daqiri.h>

#include <jetstream/backend/base.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/registry.hh>

#include "../endpoint.hh"
#include "../multicast.hh"

#include "module_impl.hh"
#include "detail/block.hh"
#include "detail/daqiri_config.hh"
#include "detail/geometry.hh"
#include "detail/kernel.hh"
#include "detail/packet.hh"
#include "detail/scatter_pool.hh"

namespace Jetstream::Modules {

using namespace stelline::domains::stelline::utils;

namespace {

class ThroughputMeter {
 public:
    bool sample(const U64 totalPackets, const F64 previousGbps, F64& gbps) {
        const auto now = std::chrono::steady_clock::now();
        const F64 elapsedSeconds = std::chrono::duration<F64>(now - lastUpdate).count();
        if (elapsedSeconds < 0.10) {
            return false;
        }

        const U64 deltaPackets = totalPackets >= lastPackets
                                     ? totalPackets - lastPackets
                                     : 0;
        const F64 instantGbps = static_cast<F64>(deltaPackets) *
                                static_cast<F64>(kPacketDataSize) *
                                8.0 / elapsedSeconds / 1.0e9;

        constexpr F64 kEmaAlpha = 0.3;
        gbps = kEmaAlpha * instantGbps + (1.0 - kEmaAlpha) * previousGbps;

        lastPackets = totalPackets;
        lastUpdate = now;
        return true;
    }

 private:
    U64 lastPackets = 0;
    std::chrono::steady_clock::time_point lastUpdate = std::chrono::steady_clock::now();
};

}  // namespace

struct AtaReceiverImplNativeCuda : public AtaReceiverImpl,
                                   public NativeCudaRuntimeContext,
                                   public Scheduler::Context {
 public:
    Result create() final;
    Result destroy() final;
    Result hasPendingCompute() override;
    Result computeSubmit(const cudaStream_t& stream) override;

 private:
    struct StreamStatsBatch {
        std::set<U64> allAntennas;
        std::set<U64> filteredAntennas;
        std::set<U64> allChannels;
        std::set<U64> filteredChannels;
        std::set<U64> payloadSizes;
    };

    Result receiveLoop();
    Result burstCollectorLoop();

    Result processBurst(const std::shared_ptr<daqiri::BurstParams>& burst);

    bool acceptPacket(const VoltagePacket& packet,
                      const AtaReceiverBlockGeometry& totalGeometry,
                      U64& antennaIndex,
                      U64& channelIndex);

    // Leaves `block` null when out of capacity so the caller drops the packet.
    Result ensureBlock(const U64 blockTimeIndex,
                       const U64 timestamp,
                       std::shared_ptr<AtaReceiverBlock>& block,
                       std::vector<std::shared_ptr<AtaReceiverBlock>>& deferredDrops);

    Result sealBlock(const std::shared_ptr<AtaReceiverBlock>& block, const bool emit);

    Result releaseReceivedBlocks();
    Result releaseComputedBlocks();
    Result updateBlockQueueDepths();

    void publishStreamStats();

    ScatterStagingPool scatterPool;
    StreamStatsBatch streamStats;
    U64 outputElementSize = 0;

    bool daqiriActive = false;

    MulticastMembership multicastMembership;
};

Result AtaReceiverImplNativeCuda::create() {
    JST_CHECK(AtaReceiverImpl::create());

    // Initialize variables.

    errored.store(false);

    std::vector<SubscriptionEndpoint> parsedSubscriptions;
    JST_CHECK(ParseSubscriptions(subscriptions, parsedSubscriptions));

    // Set CUDA device.

    JST_CUDA_CHECK(cudaSetDevice(static_cast<int>(gpuDeviceId)), [&] {
        JST_ERROR("[MODULE_ATA_RECEIVER_NATIVE_CUDA] Failed to set CUDA device: {}", err);
    });

    // Initialize DAQIRI.

    {
        // Unified memory platforms (integrated GPUs, e.g. DGX Spark) take the
        // payload regions in pinned host memory instead: same physical DRAM,
        // without depending on GPUDirect registration of device memory.
        const bool unifiedMemory = Backend::State<DeviceType::CUDA>()->hasUnifiedMemory();
        if (unifiedMemory) {
            JST_INFO("[MODULE_ATA_RECEIVER_NATIVE_CUDA] Unified memory platform detected. "
                     "Using pinned host memory for packet payloads.");
        }

        DaqiriRxConfigParams params = {};
        params.interfaceAddress = interfaceAddress;
        params.gpuDeviceId = gpuDeviceId;
        params.masterCore = masterCore;
        params.workerCores = workerCores;
        params.packetsPerBurst = packetsPerBurst;
        params.maxConcurrentBursts = maxConcurrentBursts;
        params.dataMemoryKind = unifiedMemory ? daqiri::MemoryKind::HOST_PINNED
                                              : daqiri::MemoryKind::DEVICE;

        daqiri::NetworkConfig cfg = {};
        JST_CHECK(BuildDaqiriRxConfig(params, parsedSubscriptions, cfg));

        const auto initStatus = daqiri::daqiri_init(cfg);
        if (initStatus != daqiri::Status::SUCCESS) {
            JST_ERROR("[MODULE_ATA_RECEIVER_NATIVE_CUDA] Failed to configure the DAQIRI engine.");
            if (initStatus == daqiri::Status::INTERNAL_ERROR) {
                daqiri::shutdown();
            }
            return Result::ERROR;
        }
        daqiriActive = true;
    }

    // Allocate reception blocks.

    for (U64 blockIndex = 0; blockIndex < maxConcurrentBlocks; blockIndex++) {
        idleQueue.push(std::make_shared<AtaReceiverBlock>(packetsPerBlock));
    }

    JST_CHECK(updateBlockQueueDepths());

    // Allocate output tensors.

    for (U64 poolIndex = 0; poolIndex < outputPoolSize; poolIndex++) {
        auto tensor = std::make_shared<Tensor>();
        JST_CHECK(tensor->create(DeviceType::CUDA, NameToDataType(dataType), totalBlock));
        JST_CHECK(tensor->setAttribute("timestamp", static_cast<U64>(0)));
        availableOutputTensors.push(tensor);
    }
    outputPoolDepth.publish(availableOutputTensors.size());

    outputElementSize = (NameToDataType(dataType) == DataType::CI8) ? 2 : 8;

    JST_CHECK(scatterPool.create(maxConcurrentBursts, packetsPerBurst));

    JST_CHECK(multicastMembership.create(interfaceAddress, parsedSubscriptions));

    // Flush receiving queues.

    daqiri::drop_all_traffic(0);

    // Start all threads.

    packetProcessingThreadRunning.store(true);
    burstCollectorThreadRunning.store(true);

    const auto handleThreadFailure = [this]() {
        errored.store(true);
        packetProcessingThreadRunning.store(false);
        burstCollectorThreadRunning.store(false);
    };

    packetProcessingThread = std::thread([this, handleThreadFailure]() {
        try {
            JST_CHECK_THROW(receiveLoop());
        } catch (const Result&) {
            handleThreadFailure();
        } catch (const std::exception& error) {
            JST_ERROR("[MODULE_ATA_RECEIVER_NATIVE_CUDA] {}", error.what());
            handleThreadFailure();
        }
    });

    burstCollectorThread = std::thread([this, handleThreadFailure]() {
        try {
            JST_CHECK_THROW(burstCollectorLoop());
        } catch (const Result&) {
            handleThreadFailure();
        } catch (const std::exception& error) {
            JST_ERROR("[MODULE_ATA_RECEIVER_NATIVE_CUDA] {}", error.what());
            handleThreadFailure();
        }
    });

    return Result::SUCCESS;
}

Result AtaReceiverImplNativeCuda::destroy() {
    stopThreads();

    scatterPool.drain();

    const auto result = AtaReceiverImpl::destroy();

    scatterPool.destroy();

    if (daqiriActive) {
        daqiri::shutdown();
        daqiriActive = false;
    }

    multicastMembership.destroy();
    return result;
}

Result AtaReceiverImplNativeCuda::hasPendingCompute() {
    if (errored.load()) {
        return Result::ERROR;
    }

    std::lock_guard<std::mutex> lock(readyMutex);
    if (readyOutputTensors.empty()) {
        return Result::YIELD;
    }

    return Result::SUCCESS;
}

Result AtaReceiverImplNativeCuda::computeSubmit(const cudaStream_t&) {
    if (errored.load()) {
        return Result::ERROR;
    }

    AtaReceiverReadyTensor ready;
    JST_CHECK(popReadyTensor(ready));

    const auto& swapResult = outputTensor.swapBuffers(*ready.tensor);

    if (swapResult != Result::SUCCESS) {
        JST_CHECK(recycleOutputTensor(ready.tensor));
        return swapResult;
    }

    JST_CHECK(outputTensor.setAttribute("timestamp", ready.timestamp));
    JST_CHECK(recycleOutputTensor(ready.tensor));

    emittedBlocks.publish(emittedBlocks.get() + 1);

    return Result::SUCCESS;
}

Result AtaReceiverImplNativeCuda::receiveLoop() {
    JST_CUDA_CHECK(cudaSetDevice(static_cast<int>(gpuDeviceId)), [&] {
        JST_ERROR("[MODULE_ATA_RECEIVER_NATIVE_CUDA] Failed to set CUDA device for ATA receive thread: {}", err);
    });

    ThroughputMeter throughputMeter;

    daqiri::allow_all_traffic(0);

    while (packetProcessingThreadRunning.load()) {
        daqiri::BurstParams* burstPointer = nullptr;

        if (daqiri::get_rx_burst(&burstPointer) == daqiri::Status::SUCCESS) {
            auto burst = std::shared_ptr<daqiri::BurstParams>(burstPointer, [](daqiri::BurstParams*) {});
            {
                std::lock_guard<std::mutex> lock(burstCollectorMutex);
                bursts.insert(burst);
                burstsInFlight.publish(bursts.size());
            }

            JST_CHECK(processBurst(burst));
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }

        JST_CHECK(scatterPool.reap(false));
        JST_CHECK(releaseReceivedBlocks());
        JST_CHECK(releaseComputedBlocks());

        const U64 totalPackets = receivedPackets.get() +
                                 evictedPackets.get() +
                                 lostPackets.get();
        F64 gbps = 0.0;
        if (throughputMeter.sample(totalPackets, throughputGbps.get(), gbps)) {
            throughputGbps.publish(gbps);
            publishStreamStats();
        }
    }

    return Result::SUCCESS;
}

Result AtaReceiverImplNativeCuda::burstCollectorLoop() {
    std::chrono::microseconds totalRuntime(0);
    U64 numIterations = 1;

    while (burstCollectorThreadRunning.load()) {
        const auto startTime = std::chrono::steady_clock::now();
        std::unordered_set<std::shared_ptr<daqiri::BurstParams>> staleBursts;

        {
            std::lock_guard<std::mutex> lock(burstCollectorMutex);
            for (const auto& burst : bursts) {
                if (burst.unique()) {
                    staleBursts.insert(burst);
                }
            }

            for (const auto& burst : staleBursts) {
                bursts.erase(burst);
            }

            burstsInFlight.publish(bursts.size());
        }

        for (const auto& burst : staleBursts) {
            daqiri::free_all_packets_and_burst_rx(burst.get());
        }

        const auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - startTime);
        totalRuntime += elapsedTime;
        numIterations += staleBursts.size();
        avgBurstReleaseTimeUs.publish((totalRuntime / numIterations).count());

        const auto sleepTime = std::chrono::microseconds(5000) - elapsedTime;
        if (sleepTime.count() > 0) {
            std::this_thread::sleep_for(sleepTime);
        }
    }

    return Result::SUCCESS;
}

Result AtaReceiverImplNativeCuda::processBurst(const std::shared_ptr<daqiri::BurstParams>& burst) {
    const AtaReceiverBlockGeometry totalGeometry{
        totalBlock[kAntennaAxis],
        totalBlock[kChannelAxis],
        totalBlock[kSampleAxis],
        totalBlock[kPolarizationAxis],
    };
    const AtaReceiverBlockGeometry partialGeometry{
        partialBlock[kAntennaAxis],
        partialBlock[kChannelAxis],
        partialBlock[kSampleAxis],
        partialBlock[kPolarizationAxis],
    };
    ScatterStagingPool::Staging* staging = nullptr;
    JST_CHECK(scatterPool.acquire(staging));
    U64 stagedCount = 0;
    std::vector<std::shared_ptr<AtaReceiverBlock>> deferredDrops;

    for (int64_t burstPacketIndex = 0; burstPacketIndex < daqiri::get_num_packets(burst.get()); burstPacketIndex++) {
        const auto* packetHeader = reinterpret_cast<const U8*>(
            daqiri::get_segment_packet_ptr(burst.get(), kRxHeaderSegment, burstPacketIndex));
        const VoltagePacket packet(packetHeader + kPacketHeaderOffset);

        U64 antennaIndex = packet.antennaId;
        U64 channelIndex = packet.channelNumber;

        streamStats.allAntennas.insert(antennaIndex);
        streamStats.allChannels.insert(channelIndex);
        streamStats.payloadSizes.insert(daqiri::get_segment_packet_length(burst.get(), kRxDataSegment, burstPacketIndex));

        if (!acceptPacket(packet, totalGeometry, antennaIndex, channelIndex)) {
            continue;
        }

        const U64 blockTimeIndex = packet.timestamp / blockDuration;
        const U64 blockPacketTimeIndex = (packet.timestamp % blockDuration) / partialGeometry.numberOfSamples;
        if (blockTimeIndex > latestBlockTimeIndex.get()) {
            latestBlockTimeIndex.publish(blockTimeIndex);
        }

        std::shared_ptr<AtaReceiverBlock> block;
        JST_CHECK(ensureBlock(blockTimeIndex, packet.timestamp, block, deferredDrops));
        if (!block) {
            continue;
        }

        antennaIndex /= partialGeometry.numberOfAntennas;
        channelIndex /= partialGeometry.numberOfChannels;

        streamStats.filteredAntennas.insert(antennaIndex);
        streamStats.filteredChannels.insert(channelIndex);

        if (stagedCount >= packetsPerBurst) {
            lostPackets.publish(lostPackets.get() + 1);
            continue;
        }

        const U64 blockPacketIndex = PacketSlotIndex(totalGeometry, partialGeometry,
                                                     antennaIndex, channelIndex, blockPacketTimeIndex);
        if (!block->claimSlot(blockPacketIndex)) {
            evictedPackets.publish(evictedPackets.get() + 1);
            continue;
        }

        const U64 elementOffset = PacketElementOffset(totalGeometry, partialGeometry,
                                                      antennaIndex, channelIndex, blockPacketTimeIndex);

        staging->sources[stagedCount] =
            daqiri::get_segment_packet_ptr(burst.get(), kRxDataSegment, burstPacketIndex);
        staging->destinations[stagedCount] =
            static_cast<U8*>(block->outputTensor->data()) + elementOffset * outputElementSize;
        stagedCount++;

        receivedPackets.publish(receivedPackets.get() + 1);
    }

    if (stagedCount > 0) {
        JST_CUDA_CHECK(LaunchAtaScatterKernel(staging->sources,
                                              staging->destinations,
                                              stagedCount,
                                              totalGeometry,
                                              partialGeometry,
                                              dataType,
                                              scatterPool.stream()), [&] {
            JST_ERROR("[MODULE_ATA_RECEIVER_NATIVE_CUDA] Failed to launch ATA scatter kernel: {}", err);
        });
        JST_CHECK(scatterPool.submit(staging, burst));
    } else {
        scatterPool.release(staging);
    }

    for (const auto& block : deferredDrops) {
        JST_CHECK(sealBlock(block, false));
    }

    return Result::SUCCESS;
}

bool AtaReceiverImplNativeCuda::acceptPacket(const VoltagePacket& packet,
                                             const AtaReceiverBlockGeometry& totalGeometry,
                                             U64& antennaIndex,
                                             U64& channelIndex) {
    antennaIndex -= offsetBlock[kAntennaAxis];
    channelIndex -= offsetBlock[kChannelAxis];

    if (antennaIndex >= totalGeometry.numberOfAntennas) {
        evictedPackets.publish(evictedPackets.get() + 1);
        return false;
    }

    if (channelIndex >= totalGeometry.numberOfChannels) {
        evictedPackets.publish(evictedPackets.get() + 1);
        return false;
    }

    if (packet.timestamp > latestTimestamp.get()) {
        latestTimestamp.publish(packet.timestamp);
    }

    if (packet.timestamp < timestampCutoff) {
        evictedPackets.publish(evictedPackets.get() + 1);
        return false;
    }

    return true;
}

Result AtaReceiverImplNativeCuda::ensureBlock(const U64 blockTimeIndex,
                                              const U64 timestamp,
                                              std::shared_ptr<AtaReceiverBlock>& block,
                                              std::vector<std::shared_ptr<AtaReceiverBlock>>& deferredDrops) {
    if (blockMap.contains(blockTimeIndex)) {
        block = blockMap.at(blockTimeIndex);
        return Result::SUCCESS;
    }

    std::shared_ptr<Tensor> tensor;
    if (!idleQueue.empty()) {
        tensor = tryAcquireOutputTensor();
    }

    if (idleQueue.empty() || !tensor) {
        if (!receiveQueue.empty()) {
            auto oldest = receiveQueue.front();
            receiveQueue.pop();
            blockMap.erase(oldest->index);
            // Current burst may already have staged writes to this block.
            // Record its completion event only after the burst scatter launch.
            deferredDrops.push_back(oldest);
            lostBlocks.publish(lostBlocks.get() + 1);
        }
        lostPackets.publish(lostPackets.get() + 1);
        block = nullptr;
        return Result::SUCCESS;
    }

    block = idleQueue.front();
    idleQueue.pop();
    block->create(blockTimeIndex, timestamp, tensor);
    receiveQueue.push(block);
    blockMap[blockTimeIndex] = block;
    JST_CHECK(updateBlockQueueDepths());

    return Result::SUCCESS;
}

Result AtaReceiverImplNativeCuda::sealBlock(const std::shared_ptr<AtaReceiverBlock>& block, const bool emit) {
    block->emit = emit;
    JST_CUDA_CHECK(cudaEventRecord(block->event, scatterPool.stream()), [&] {
        JST_ERROR("[MODULE_ATA_RECEIVER_NATIVE_CUDA] Failed to record block event: {}", err);
    });
    computeQueue.push(block);

    return Result::SUCCESS;
}

Result AtaReceiverImplNativeCuda::releaseReceivedBlocks() {
    while (!receiveQueue.empty()) {
        auto block = receiveQueue.front();
        receiveQueue.pop();

        const bool isStale = latestBlockTimeIndex.get() > (block->index + maxConcurrentBlocks);

        if (block->isComplete()) {
            blockMap.erase(block->index);
            JST_CHECK(sealBlock(block, true));
            receivedBlocks.publish(receivedBlocks.get() + 1);
        } else if (isStale) {
            blockMap.erase(block->index);
            JST_CHECK(sealBlock(block, false));
            lostBlocks.publish(lostBlocks.get() + 1);
        } else {
            swapQueue.push(block);
        }
    }

    while (!swapQueue.empty()) {
        receiveQueue.push(swapQueue.front());
        swapQueue.pop();
    }

    JST_CHECK(updateBlockQueueDepths());

    return Result::SUCCESS;
}

Result AtaReceiverImplNativeCuda::releaseComputedBlocks() {
    while (!computeQueue.empty()) {
        auto block = computeQueue.front();

        // Block events share the scatter stream, so they complete in seal
        // order: once the front is pending, everything behind it is too.
        const auto status = cudaEventQuery(block->event);
        if (status == cudaErrorNotReady) {
            break;
        }
        if (status != cudaSuccess) {
            JST_ERROR("[MODULE_ATA_RECEIVER_NATIVE_CUDA] ATA scatter kernel failed on the block stream: {}",
                      cudaGetErrorString(status));
            return Result::ERROR;
        }

        computeQueue.pop();

        if (block->emit) {
            auto tensor = block->outputTensor;
            JST_CHECK(tensor->setAttribute("timestamp", block->timestamp));
            JST_CHECK(pushReadyTensor(tensor, block->timestamp));
        } else {
            JST_CHECK(recycleOutputTensor(block->outputTensor));
        }
        block->destroy();
        idleQueue.push(block);
    }

    JST_CHECK(updateBlockQueueDepths());

    return Result::SUCCESS;
}

Result AtaReceiverImplNativeCuda::updateBlockQueueDepths() {
    idleQueueDepth.publish(idleQueue.size());
    receiveQueueDepth.publish(receiveQueue.size());
    computeQueueDepth.publish(computeQueue.size());
    blockMapDepth.publish(blockMap.size());

    return Result::SUCCESS;
}

void AtaReceiverImplNativeCuda::publishStreamStats() {
    allAntennas.publish(streamStats.allAntennas);
    filteredAntennas.publish(streamStats.filteredAntennas);
    allChannels.publish(streamStats.allChannels);
    filteredChannels.publish(streamStats.filteredChannels);
    payloadSizes.publish(streamStats.payloadSizes);
}

JST_REGISTER_MODULE(AtaReceiverImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
