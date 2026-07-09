#include "module_impl.hh"

#include "../endpoint.hh"

#include "detail/block.hh"

namespace Jetstream::Modules {

using namespace stelline::domains::stelline::utils;

namespace {

bool ValidateBlockShape(const std::vector<U64>& shape,
                        const std::string& label,
                        const bool allowZero) {
    if (shape.size() != kShapeRank) {
        JST_ERROR("[MODULE_ATA_RECEIVER] {} must have 4 dimensions [A, C, S, P].", label);
        return false;
    }

    for (const auto& value : shape) {
        if (!allowZero && value == 0) {
            JST_ERROR("[MODULE_ATA_RECEIVER] {} contains an invalid dimension.", label);
            return false;
        }
    }

    return true;
}

}  // namespace

Result AtaReceiverImpl::validate() {
    const auto& config = *candidate();

    if (config.engine != "ibverbs") {
        JST_ERROR("[MODULE_ATA_RECEIVER] Unsupported engine '{}'. Valid values: ibverbs.", config.engine);
        return Result::ERROR;
    }

    if (!ValidateBlockShape(config.totalBlock, "totalBlock", false)) {
        return Result::ERROR;
    }

    if (!ValidateBlockShape(config.partialBlock, "partialBlock", false)) {
        return Result::ERROR;
    }

    if (!ValidateBlockShape(config.offsetBlock, "offsetBlock", true)) {
        return Result::ERROR;
    }

    if (config.offsetBlock[kSampleAxis] != 0 || config.offsetBlock[kPolarizationAxis] != 0) {
        JST_ERROR("[MODULE_ATA_RECEIVER] The 'offsetBlock' samples and polarizations must be zero.");
        return Result::ERROR;
    }

    if ((config.offsetBlock[kAntennaAxis] % config.partialBlock[kAntennaAxis]) != 0) {
        JST_ERROR("[MODULE_ATA_RECEIVER] The 'offsetBlock' antennas must align with 'partialBlock' antennas.");
        return Result::ERROR;
    }

    for (U64 axis = 0; axis < kShapeRank; axis++) {
        if ((config.totalBlock[axis] % config.partialBlock[axis]) != 0) {
            JST_ERROR("[MODULE_ATA_RECEIVER] The 'totalBlock' must be divisible by 'partialBlock' on axis {}.", axis);
            return Result::ERROR;
        }
    }

    if (config.maxConcurrentBlocks == 0) {
        JST_ERROR("[MODULE_ATA_RECEIVER] The 'maxConcurrentBlocks' must be positive.");
        return Result::ERROR;
    }

    if (config.outputPoolSize == 0) {
        JST_ERROR("[MODULE_ATA_RECEIVER] The 'outputPoolSize' must be positive.");
        return Result::ERROR;
    }

    if (NameToDataType(config.dataType) != DataType::CF32 &&
        NameToDataType(config.dataType) != DataType::CI8) {
        JST_ERROR("[MODULE_ATA_RECEIVER] Unsupported data type '{}'.", config.dataType);
        return Result::ERROR;
    }

    if (config.interfaceAddress.empty()) {
        JST_ERROR("[MODULE_ATA_RECEIVER] The 'interfaceAddress' must not be empty.");
        return Result::ERROR;
    }

    if (config.workerCores.empty()) {
        JST_ERROR("[MODULE_ATA_RECEIVER] The 'workerCores' must contain at least one core.");
        return Result::ERROR;
    }

    if (config.packetsPerBurst == 0) {
        JST_ERROR("[MODULE_ATA_RECEIVER] The 'packetsPerBurst' must be positive.");
        return Result::ERROR;
    }

    if (config.maxConcurrentBursts == 0) {
        JST_ERROR("[MODULE_ATA_RECEIVER] The 'maxConcurrentBursts' must be positive.");
        return Result::ERROR;
    }

    if (config.subscriptions.empty()) {
        JST_ERROR("[MODULE_ATA_RECEIVER] The 'subscriptions' must not be empty.");
        return Result::ERROR;
    }

    std::vector<SubscriptionEndpoint> parsedSubscriptions;
    if (ParseSubscriptions(config.subscriptions, parsedSubscriptions) != Result::SUCCESS) {
        return Result::ERROR;
    }

    for (const auto& subscription : parsedSubscriptions) {
        EndpointMatch source;
        if (ParseEndpoint(subscription.source, source) != Result::SUCCESS) {
            return Result::ERROR;
        }

        EndpointMatch destination;
        if (ParseEndpoint(subscription.destination, destination) != Result::SUCCESS) {
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

Result AtaReceiverImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::CLEAN));
    JST_CHECK(defineInterfaceOutput("output"));

    return Result::SUCCESS;
}

Result AtaReceiverImpl::create() {
    slotShape = {
        totalBlock[kAntennaAxis] / partialBlock[kAntennaAxis],
        totalBlock[kChannelAxis] / partialBlock[kChannelAxis],
        totalBlock[kSampleAxis] / partialBlock[kSampleAxis],
        totalBlock[kPolarizationAxis] / partialBlock[kPolarizationAxis],
    };

    packetsPerBlock = 1;
    for (const auto& value : slotShape) {
        packetsPerBlock *= value;
    }

    packetDuration = partialBlock[kSampleAxis];
    blockDuration = slotShape[kSampleAxis] * packetDuration;

    timestampCutoff = 0;

    latestBlockTimeIndex.publish(0);
    latestTimestamp.publish(0);
    blockMapDepth.publish(0);

    receivedPackets.publish(0);
    evictedPackets.publish(0);
    lostPackets.publish(0);
    receivedBlocks.publish(0);
    emittedBlocks.publish(0);
    lostBlocks.publish(0);

    idleQueueDepth.publish(0);
    receiveQueueDepth.publish(0);
    computeQueueDepth.publish(0);
    readyQueueDepth.publish(0);
    outputPoolDepth.publish(0);
    burstsInFlight.publish(0);
    avgBurstReleaseTimeUs.publish(0);
    throughputGbps.publish(0.0);

    allAntennas.publish({});
    filteredAntennas.publish({});
    allChannels.publish({});
    filteredChannels.publish({});
    payloadSizes.publish({});

    JST_CHECK(outputTensor.create(device(), NameToDataType(dataType), totalBlock));
    JST_CHECK(outputTensor.setAttribute("timestamp", static_cast<U64>(0)));

    outputs()["output"].produced(name(), "output", outputTensor);

    return Result::SUCCESS;
}

Result AtaReceiverImpl::destroy() {
    stopThreads();

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
        std::unordered_set<std::shared_ptr<daqiri::BurstParams>> staleBursts;
        {
            std::lock_guard<std::mutex> lock(burstCollectorMutex);
            staleBursts.swap(bursts);
            burstsInFlight.publish(0);
        }

        for (const auto& burst : staleBursts) {
            daqiri::free_all_packets_and_burst_rx(burst.get());
        }
    }

    return Result::SUCCESS;
}

Result AtaReceiverImpl::reconfigure() {
    return Result::RECREATE;
}

void AtaReceiverImpl::stopThreads() {
    packetProcessingThreadRunning.store(false);
    burstCollectorThreadRunning.store(false);

    if (packetProcessingThread.joinable()) {
        packetProcessingThread.join();
    }

    if (burstCollectorThread.joinable()) {
        burstCollectorThread.join();
    }
}

Result AtaReceiverImpl::pushReadyTensor(const std::shared_ptr<Tensor>& tensor, const U64 timestamp) {
    std::lock_guard<std::mutex> lock(readyMutex);
    readyOutputTensors.push({tensor, timestamp});
    readyQueueDepth.publish(readyOutputTensors.size());

    return Result::SUCCESS;
}

Result AtaReceiverImpl::popReadyTensor(AtaReceiverReadyTensor& ready) {
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

std::shared_ptr<Tensor> AtaReceiverImpl::tryAcquireOutputTensor() {
    std::lock_guard<std::mutex> lock(poolMutex);
    if (availableOutputTensors.empty()) {
        return nullptr;
    }

    auto tensor = availableOutputTensors.front();
    availableOutputTensors.pop();
    outputPoolDepth.publish(availableOutputTensors.size());
    return tensor;
}

Result AtaReceiverImpl::recycleOutputTensor(const std::shared_ptr<Tensor>& tensor) {
    std::lock_guard<std::mutex> lock(poolMutex);
    availableOutputTensors.push(tensor);
    outputPoolDepth.publish(availableOutputTensors.size());

    return Result::SUCCESS;
}

U64 AtaReceiverImpl::getReceivedBlocks() const {
    return receivedBlocks.get();
}

U64 AtaReceiverImpl::getComputedBlocks() const {
    return emittedBlocks.get();
}

U64 AtaReceiverImpl::getEmittedBlocks() const {
    return emittedBlocks.get();
}

U64 AtaReceiverImpl::getLostBlocks() const {
    return lostBlocks.get();
}

U64 AtaReceiverImpl::getReceivedPackets() const {
    return receivedPackets.get();
}

U64 AtaReceiverImpl::getEvictedPackets() const {
    return evictedPackets.get();
}

U64 AtaReceiverImpl::getLostPackets() const {
    return lostPackets.get();
}

U64 AtaReceiverImpl::getLatestTimestamp() const {
    return latestTimestamp.get();
}

F64 AtaReceiverImpl::getInputGbps() const {
    return throughputGbps.get();
}

U64 AtaReceiverImpl::getIdleQueue() const {
    return idleQueueDepth.get();
}

U64 AtaReceiverImpl::getMaxConcurrentBursts() const {
    return maxConcurrentBursts;
}

U64 AtaReceiverImpl::getAverageBurstReleaseTimeUs() const {
    return avgBurstReleaseTimeUs.get();
}

U64 AtaReceiverImpl::getBurstsInFlight() const {
    return burstsInFlight.get();
}

U64 AtaReceiverImpl::getMemPoolAvailable() const {
    return outputPoolDepth.get();
}

U64 AtaReceiverImpl::getMemPoolReferenced() const {
    const U64 available = outputPoolDepth.get();
    return available <= outputPoolSize ? outputPoolSize - available : 0;
}

U64 AtaReceiverImpl::getBlockMapLatestTimeIndex() const {
    return latestBlockTimeIndex.get();
}

U64 AtaReceiverImpl::getBlockMapUsed() const {
    return blockMapDepth.get();
}

U64 AtaReceiverImpl::getBlockMapCapacity() const {
    return maxConcurrentBlocks;
}

U64 AtaReceiverImpl::getReceiveQueue() const {
    return receiveQueueDepth.get();
}

U64 AtaReceiverImpl::getComputeQueue() const {
    return computeQueueDepth.get();
}

U64 AtaReceiverImpl::getReadyQueue() const {
    return readyQueueDepth.get();
}

std::string AtaReceiverImpl::getPayloadSizes() const {
    return jst::fmt::format("{}", payloadSizes.get());
}

std::string AtaReceiverImpl::getAllAntennas() const {
    return jst::fmt::format("{}", allAntennas.get());
}

std::string AtaReceiverImpl::getFilteredAntennas() const {
    return jst::fmt::format("{}", filteredAntennas.get());
}

std::string AtaReceiverImpl::getAllChannels() const {
    return jst::fmt::format("{}", allChannels.get());
}

std::string AtaReceiverImpl::getFilteredChannels() const {
    return jst::fmt::format("{}", filteredChannels.get());
}

}  // namespace Jetstream::Modules
