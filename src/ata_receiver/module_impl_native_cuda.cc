#include <arpa/inet.h>

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>

#include "jetstream/backend/devices/cuda/helpers.hh"

#include <advanced_network/common.h>

#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/registry.hh>

#include "../net/endpoint.hh"
#include "../net/multicast.hh"

#include "detail/kernel.hh"
#include "module_impl.hh"
#include "detail/packet.hh"
#include "detail/block.hh"

namespace Jetstream::Modules {

namespace ano = holoscan::advanced_network;
using namespace stelline::domains::stelline::utils;

struct AtaReceiverImplNativeCuda : public AtaReceiverImpl,
                                   public NativeCudaRuntimeContext,
                                   public Scheduler::Context {
 public:
    Result create() final;
    Result destroy() final;
    Result hasPendingCompute() override;
    Result computeSubmit(const cudaStream_t& stream) override;

  private:
    Result receiveLoop();
    Result burstCollectorLoop();

    Result processBurst(const std::shared_ptr<ano::BurstParams>& burst);

    Result releaseReceivedBlocks();
    Result releaseComputedBlocks();
    Result updateBlockQueueDepths();

    Result pushReadyTensor(const std::shared_ptr<Tensor>& tensor, const U64 timestamp);
    Result popReadyTensor(AtaReceiverReadyTensor& ready);
    Result recycleOutputTensor(const std::shared_ptr<Tensor>& tensor);

    std::shared_ptr<Tensor> acquireOutputTensor();

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

    // Initialize Advanced Network

    {
        const auto subscriptionCount = parsedSubscriptions.size();
        const auto totalNumBufs = static_cast<size_t>(packetsPerBurst * maxConcurrentBursts);

        ano::NetworkConfig cfg = {};
        cfg.log_level_ = ano::LogLevel::TRACE;
        cfg.tx_meta_buffers_ = ano::DEFAULT_TX_META_BUFFERS * 8;
        cfg.rx_meta_buffers_ = ano::DEFAULT_RX_META_BUFFERS * 8;

        cfg.common_.version = 1;
        cfg.common_.master_core_ = masterCore;
        cfg.common_.dir = ano::Direction::RX;
        cfg.common_.manager_type = ano::ManagerType::DPDK;
        cfg.common_.loopback_ = ano::LoopbackType::DISABLED;

        ano::InterfaceConfig interfaceCfg = {};
        interfaceCfg.address_ = interfaceAddress;
        interfaceCfg.rx_.flow_isolation_ = true;

        U16 nextId = 0;
        for (const auto& subscription : parsedSubscriptions) {
            const auto queueNumBufs = totalNumBufs / subscriptionCount +
                                      (static_cast<size_t>(nextId) < (totalNumBufs % subscriptionCount) ? 1 : 0);
            if (queueNumBufs == 0) {
                JST_ERROR("[MODULE_ATA_RECEIVER_NATIVE_CUDA] Total buffer count is too small for the number of subscriptions.");
                return Result::ERROR;
            }

            const auto headerMrName = "RX_HEADER_" + std::to_string(nextId);
            const auto dataMrName = "RX_DATA_" + std::to_string(nextId);

            ano::MemoryRegionConfig headerMemoryCfg = {};
            headerMemoryCfg.name_ = headerMrName;
            headerMemoryCfg.kind_ = ano::MemoryKind::HUGE;
            headerMemoryCfg.affinity_ = 0;
            headerMemoryCfg.buf_size_ = kPacketHeaderOffset + kPacketHeaderSize;
            headerMemoryCfg.num_bufs_ = queueNumBufs;
            headerMemoryCfg.access_ = ano::MEM_ACCESS_LOCAL;
            headerMemoryCfg.owned_ = true;
            cfg.mrs_.emplace(headerMemoryCfg.name_, headerMemoryCfg);

            ano::MemoryRegionConfig dataMemoryCfg = {};
            dataMemoryCfg.name_ = dataMrName;
            dataMemoryCfg.kind_ = ano::MemoryKind::DEVICE;
            dataMemoryCfg.affinity_ = gpuDeviceId;
            dataMemoryCfg.buf_size_ = kPacketDataSize;
            dataMemoryCfg.num_bufs_ = queueNumBufs;
            dataMemoryCfg.access_ = ano::MEM_ACCESS_LOCAL;
            dataMemoryCfg.owned_ = true;
            cfg.mrs_.emplace(dataMemoryCfg.name_, dataMemoryCfg);

            ano::RxQueueConfig queueCfg = {};
            queueCfg.common_.name_ = "subscription-" + std::to_string(nextId);
            queueCfg.common_.id_ = nextId;
            queueCfg.common_.batch_size_ = packetsPerBurst;
            queueCfg.common_.cpu_core_ = jst::fmt::format("{}", workerCores[nextId % workerCores.size()]);
            queueCfg.common_.mrs_ = {headerMrName, dataMrName};
            queueCfg.timeout_us_ = 0;
            interfaceCfg.rx_.queues_.push_back(queueCfg);

            EndpointMatch source;
            JST_CHECK(ParseEndpoint(subscription.source, source));

            EndpointMatch destination;
            JST_CHECK(ParseEndpoint(subscription.destination, destination));

            ano::FlowConfig flowCfg = {};
            flowCfg.name_ = "subscription-" + std::to_string(nextId);
            flowCfg.id_ = nextId;
            flowCfg.action_.type_ = ano::FlowType::QUEUE;
            flowCfg.action_.id_ = nextId;
            flowCfg.match_.type_ = ano::FlowMatchType::NORMAL;
            flowCfg.match_.udp_src_ = source.hasPort ? source.port : 0;
            flowCfg.match_.udp_dst_ = destination.hasPort ? destination.port : 0;
            flowCfg.match_.ipv4_len_ = 0;
            flowCfg.match_.ipv4_src_ = source.hasIp ? source.ip : INADDR_ANY;
            flowCfg.match_.ipv4_dst_ = destination.hasIp ? destination.ip : INADDR_ANY;
            interfaceCfg.rx_.flows_.push_back(flowCfg);

            nextId += 1;
        }

        cfg.ifs_.push_back(interfaceCfg);

        if (ano::adv_net_init(cfg) != ano::Status::SUCCESS) {
            JST_ERROR("[MODULE_ATA_RECEIVER_NATIVE_CUDA] Failed to configure the Advanced Network manager.");
            return Result::ERROR;
        }
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

    JST_CHECK(multicastMembership.create(interfaceAddress, parsedSubscriptions));

    // Flush receiving queues.

    ano::drop_all_traffic(0);

    // Start all threads.

    packetProcessingThreadRunning.store(true);
    burstCollectorThreadRunning.store(true);

    const auto handleThreadFailure = [this]() {
        errored.store(true);
        packetProcessingThreadRunning.store(false);
        burstCollectorThreadRunning.store(false);
        outputPoolCv.notify_all();
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
    const auto result = AtaReceiverImpl::destroy();
    multicastMembership.destroy();
    return result;
}

Result AtaReceiverImpl::destroy() {
    // Stop all threads.

    packetProcessingThreadRunning.store(false);
    burstCollectorThreadRunning.store(false);

    outputPoolCv.notify_all();

    if (packetProcessingThread.joinable()) {
        packetProcessingThread.join();
    }

    if (burstCollectorThread.joinable()) {
        burstCollectorThread.join();
    }

    // Destroy reception blocks.

    while (!idleQueue.empty()) {
        idleQueue.front()->destroy();
        idleQueue.pop();
    }

    while (!receiveQueue.empty()) {
        receiveQueue.front()->destroy();
        receiveQueue.pop();
    }

    while (!computeQueue.empty()) {
        computeQueue.front()->destroy();
        computeQueue.pop();
    }

    while (!swapQueue.empty()) {
        swapQueue.front()->destroy();
        swapQueue.pop();
    }

    // Destroy output tensors.

    for (auto& [_, block] : blockMap) {
        block->destroy();
    }
    blockMap.clear();
    blockMapDepth.publish(0);

    while (!readyOutputTensors.empty()) {
        readyOutputTensors.pop();
    }

    while (!availableOutputTensors.empty()) {
        availableOutputTensors.pop();
    }
    outputPoolDepth.publish(0);

    // Destroy bursts.

    {
        std::unordered_set<std::shared_ptr<ano::BurstParams>> staleBursts;
        {
            std::lock_guard<std::mutex> lock(burstCollectorMutex);
            staleBursts.swap(bursts);
            burstsInFlight.publish(0);
        }

        for (const auto& burst : staleBursts) {
            ano::free_all_packets_and_burst_rx(burst.get());
        }
    }

    return Result::SUCCESS;
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

    U64 lastThroughputPackets = 0;
    auto lastThroughputUpdate = std::chrono::steady_clock::now();

    ano::flush_port_queue(0, 0);
    ano::allow_all_traffic(0);

    while (packetProcessingThreadRunning.load()) {
        ano::BurstParams* burstPointer = nullptr;

        if (ano::get_rx_burst(&burstPointer) == ano::Status::SUCCESS) {
            auto burst = std::shared_ptr<ano::BurstParams>(burstPointer, [](ano::BurstParams*) {});
            {
                std::lock_guard<std::mutex> lock(burstCollectorMutex);
                bursts.insert(burst);
                burstsInFlight.publish(bursts.size());
            }

            JST_CHECK(processBurst(burst));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        JST_CHECK(releaseReceivedBlocks());
        JST_CHECK(releaseComputedBlocks());

        const auto now = std::chrono::steady_clock::now();
        const F64 elapsedSeconds = std::chrono::duration<F64>(now - lastThroughputUpdate).count();

        if (elapsedSeconds >= 0.10) {
            const U64 totalPackets = receivedPackets.get() +
                                     evictedPackets.get() +
                                     lostPackets.get();
            const U64 deltaPackets = totalPackets >= lastThroughputPackets
                                         ? totalPackets - lastThroughputPackets
                                         : 0;

            const F64 instantGbps = elapsedSeconds > 0 ? static_cast<F64>(deltaPackets) *
                                                         static_cast<F64>(kPacketDataSize) *
                                                         8.0 / elapsedSeconds / 1.0e9 : 0.0;

            constexpr F64 kEmaAlpha = 0.3;
            throughputGbps.publish(kEmaAlpha * instantGbps + (1.0 - kEmaAlpha) * throughputGbps.get());

            lastThroughputPackets = totalPackets;
            lastThroughputUpdate = now;
        }
    }

    return Result::SUCCESS;
}

Result AtaReceiverImplNativeCuda::burstCollectorLoop() {
    std::chrono::microseconds totalRuntime(0);
    U64 numIterations = 1;

    while (burstCollectorThreadRunning.load()) {
        const auto startTime = std::chrono::steady_clock::now();
        std::unordered_set<std::shared_ptr<ano::BurstParams>> staleBursts;

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
            ano::free_all_packets_and_burst_rx(burst.get());
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

Result AtaReceiverImplNativeCuda::processBurst(const std::shared_ptr<ano::BurstParams>& burst) {
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
    auto allAntennasValue = allAntennas.get();
    auto filteredAntennasValue = filteredAntennas.get();
    auto allChannelsValue = allChannels.get();
    auto filteredChannelsValue = filteredChannels.get();
    auto payloadSizesValue = payloadSizes.get();

    for (int64_t burstPacketIndex = 0; burstPacketIndex < ano::get_num_packets(burst.get()); burstPacketIndex++) {
        const auto* packetHeader = reinterpret_cast<const U8*>(
            ano::get_segment_packet_ptr(burst.get(), kRxHeaderSegment, burstPacketIndex));
        const VoltagePacket packet(packetHeader + kPacketHeaderOffset);

        U64 antennaIndex = packet.antennaId;
        U64 channelIndex = packet.channelNumber;

        allAntennasValue.insert(antennaIndex);
        allChannelsValue.insert(channelIndex);
        payloadSizesValue.insert(get_segment_packet_length(burst.get(), kRxDataSegment, burstPacketIndex));

        antennaIndex -= offsetBlock[kAntennaAxis];
        channelIndex -= offsetBlock[kChannelAxis];

        if (antennaIndex >= totalGeometry.numberOfAntennas) {
            evictedPackets.publish(evictedPackets.get() + 1);
            continue;
        }

        if (channelIndex >= totalGeometry.numberOfChannels) {
            evictedPackets.publish(evictedPackets.get() + 1);
            continue;
        }

        if (packet.timestamp > latestTimestamp.get()) {
            latestTimestamp.publish(packet.timestamp);
        }
        if (packet.timestamp < timestampCutoff) {
            evictedPackets.publish(evictedPackets.get() + 1);
            continue;
        }

        const U64 blockTimeIndex = packet.timestamp / blockDuration;
        const U64 blockPacketTimeIndex = (packet.timestamp % blockDuration) / partialGeometry.numberOfSamples;
        if (blockTimeIndex > latestBlockTimeIndex.get()) {
            latestBlockTimeIndex.publish(blockTimeIndex);
        }

        if (!blockMap.contains(blockTimeIndex)) {
            if (idleQueue.empty()) {
                if (receiveQueue.empty()) {
                    lostPackets.publish(lostPackets.get() + 1);
                    continue;
                }

                auto block = receiveQueue.front();
                receiveQueue.pop();
                blockMap.erase(block->index);
                block->destroy();
                idleQueue.push(block);
                lostBlocks.publish(lostBlocks.get() + 1);
            }

            auto block = idleQueue.front();
            idleQueue.pop();
            block->create(blockTimeIndex, packet.timestamp);
            receiveQueue.push(block);
            blockMap[blockTimeIndex] = block;
            JST_CHECK(updateBlockQueueDepths());
        }

        antennaIndex /= partialGeometry.numberOfAntennas;
        channelIndex /= partialGeometry.numberOfChannels;

        filteredAntennasValue.insert(antennaIndex);
        filteredChannelsValue.insert(channelIndex);

        U64 blockPacketIndex = 0;
        blockPacketIndex += antennaIndex * slotShape[kChannelAxis] * slotShape[kSampleAxis];
        blockPacketIndex += channelIndex * slotShape[kSampleAxis];
        blockPacketIndex += blockPacketTimeIndex;

        blockMap.at(blockTimeIndex)->addPacket(blockPacketIndex, burstPacketIndex, burst);
        receivedPackets.publish(receivedPackets.get() + 1);
    }

    allAntennas.publish(allAntennasValue);
    filteredAntennas.publish(filteredAntennasValue);
    allChannels.publish(allChannelsValue);
    filteredChannels.publish(filteredChannelsValue);
    payloadSizes.publish(payloadSizesValue);

    return Result::SUCCESS;
}

Result AtaReceiverImplNativeCuda::releaseReceivedBlocks() {
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
    const AtaReceiverBlockGeometry slotGeometry{
        slotShape[kAntennaAxis],
        slotShape[kChannelAxis],
        slotShape[kSampleAxis],
        slotShape[kPolarizationAxis],
    };

    while (!receiveQueue.empty()) {
        auto block = receiveQueue.front();
        receiveQueue.pop();

        const bool isStale = latestBlockTimeIndex.get() > (block->index + maxConcurrentBlocks);

        if (block->isComplete()) {
            blockMap.erase(block->index);

            auto tensor = acquireOutputTensor();
            if (!tensor) {
                swapQueue.push(block);
                break;
            }

            if (block->compute(tensor,
                               totalGeometry,
                               partialGeometry,
                               slotGeometry,
                               dataType) != Result::SUCCESS) {
                JST_ERROR("[MODULE_ATA_RECEIVER_NATIVE_CUDA] Failed to launch ATA gather kernel.");
                JST_CHECK(recycleOutputTensor(tensor));
                return Result::ERROR;
            }

            computeQueue.push(block);
            receivedBlocks.publish(receivedBlocks.get() + 1);
        } else if (isStale) {
            blockMap.erase(block->index);
            block->destroy();
            idleQueue.push(block);
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
        computeQueue.pop();

        if (!block->isProcessing()) {
            auto tensor = block->outputTensor;
            JST_CHECK(tensor->setAttribute("timestamp", block->timestamp));
            JST_CHECK(pushReadyTensor(tensor, block->timestamp));
            block->destroy();
            idleQueue.push(block);
        } else {
            swapQueue.push(block);
        }
    }

    while (!swapQueue.empty()) {
        computeQueue.push(swapQueue.front());
        swapQueue.pop();
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

Result AtaReceiverImplNativeCuda::pushReadyTensor(const std::shared_ptr<Tensor>& tensor, const U64 timestamp) {
    std::lock_guard<std::mutex> lock(readyMutex);
    readyOutputTensors.push({tensor, timestamp});
    readyQueueDepth.publish(readyOutputTensors.size());

    return Result::SUCCESS;
}

Result AtaReceiverImplNativeCuda::popReadyTensor(AtaReceiverReadyTensor& ready) {
    std::lock_guard<std::mutex> lock(readyMutex);
    if (readyOutputTensors.empty()) {
        readyQueueDepth.publish(0);
        return Result::YIELD;
    }

    ready = readyOutputTensors.front();
    readyOutputTensors.pop();
    readyQueueDepth.publish(readyOutputTensors.size());
    return Result::SUCCESS;
}

std::shared_ptr<Tensor> AtaReceiverImplNativeCuda::acquireOutputTensor() {
    std::unique_lock<std::mutex> lock(poolMutex);
    outputPoolCv.wait(lock, [this]() {
        return !packetProcessingThreadRunning.load() || !availableOutputTensors.empty();
    });

    if (!packetProcessingThreadRunning.load() && availableOutputTensors.empty()) {
        outputPoolDepth.publish(0);
        return nullptr;
    }

    auto tensor = availableOutputTensors.front();
    availableOutputTensors.pop();
    outputPoolDepth.publish(availableOutputTensors.size());
    return tensor;
}

Result AtaReceiverImplNativeCuda::recycleOutputTensor(const std::shared_ptr<Tensor>& tensor) {
    std::lock_guard<std::mutex> lock(poolMutex);
    availableOutputTensors.push(tensor);
    outputPoolDepth.publish(availableOutputTensors.size());
    outputPoolCv.notify_one();

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(AtaReceiverImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
