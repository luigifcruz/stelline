#include <stelline/domains/transport/ata_receiver/block.hh>
#include <jetstream/detail/block_impl.hh>
#include <stelline/domains/transport/ata_receiver/module.hh>

#include "module_impl.hh"

namespace Jetstream::Blocks {

struct AtaReceiverImpl : public Block::Impl, public DynamicConfig<Blocks::AtaReceiver> {
    Result configure() override;
    Result define() override;
    Result create() override;

protected:
    std::shared_ptr<Modules::AtaReceiver> moduleConfig = std::make_shared<Modules::AtaReceiver>();
    Modules::AtaReceiverImpl* moduleImpl = nullptr;
};

Result AtaReceiverImpl::configure() {
    moduleConfig->gpuDeviceId = gpuDeviceId;
    moduleConfig->interfaceAddress = interfaceAddress;
    moduleConfig->masterCore = masterCore;
    moduleConfig->workerCores = workerCores;
    moduleConfig->packetsPerBurst = packetsPerBurst;
    moduleConfig->maxConcurrentBursts = maxConcurrentBursts;
    moduleConfig->subscriptions = subscriptions;
    moduleConfig->totalBlock = totalBlock;
    moduleConfig->partialBlock = partialBlock;
    moduleConfig->offsetBlock = offsetBlock;
    moduleConfig->maxConcurrentBlocks = maxConcurrentBlocks;
    moduleConfig->outputPoolSize = outputPoolSize;
    moduleConfig->dataType = dataType;

    return Result::SUCCESS;
}

Result AtaReceiverImpl::define() {
    JST_CHECK(defineInterfaceConfig("interfaceAddress",
                                    "Interface Address",
                                    "Network interface address used by the ATA receiver.",
                                    "text"));

    JST_CHECK(defineInterfaceConfig("gpuDeviceId",
                                    "GPU Device ID",
                                    "CUDA device ID used by the receiver backend.",
                                    "int:id"));

    JST_CHECK(defineInterfaceConfig("masterCore",
                                    "Master Core",
                                    "CPU core assigned to the receiver control thread.",
                                    "int:core"));

    JST_CHECK(defineInterfaceConfig("workerCores",
                                    "Worker Cores",
                                    "CPU cores assigned to networking workers.",
                                    "vector-inline:int:core"));

    JST_CHECK(defineInterfaceConfig("subscriptions",
                                    "Subscriptions",
                                    "One 'source:port -> destination:port' subscription per line.",
                                    "multiline"));

    JST_CHECK(defineInterfaceConfig("totalBlock",
                                    "Total Block",
                                    "Output block shape as [antennas, channels, samples, polarizations].",
                                    "vector-inline:int:dim"));

    JST_CHECK(defineInterfaceConfig("partialBlock",
                                    "Partial Block",
                                    "Per-packet fragment shape as [antennas, channels, samples, polarizations].",
                                    "vector-inline:int:dim"));

    JST_CHECK(defineInterfaceConfig("offsetBlock",
                                    "Offset Block",
                                    "Input offset as [antennas, channels, samples, polarizations].",
                                    "vector-inline:int:dim"));

    JST_CHECK(defineInterfaceConfig("dataType",
                                    "Data Type",
                                    "Output tensor data type.",
                                    "dropdown:CF32(CF32),CI8(CI8)"));

    JST_CHECK(defineInterfaceOutput("output",
                                    "Output",
                                    "Assembled ATA tensor output."));

    JST_CHECK(defineInterfaceConfig("packetsPerBurst",
                                    "Packets Per Burst",
                                    "Maximum packets expected in each burst.",
                                    "int:packets"));

    JST_CHECK(defineInterfaceConfig("maxConcurrentBursts",
                                    "Max Concurrent Bursts",
                                    "Maximum number of concurrent ANO bursts in flight.",
                                    "int:bursts"));

    JST_CHECK(defineInterfaceConfig("maxConcurrentBlocks",
                                    "Max Concurrent Blocks",
                                    "Maximum number of in-flight receive blocks.",
                                    "int:blocks"));

    JST_CHECK(defineInterfaceConfig("outputPoolSize",
                                    "Output Pool Size",
                                    "Number of reusable output tensors for completed blocks.",
                                    "int:buffers"));

    JST_CHECK(defineInterfaceMetric("blocksReceived",
                                    "Blocks Received",
                                    "Total completed blocks submitted to the gather kernel.",
                                    "stelline-metrics-global-number",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getReceivedBlocks() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("blocksComputed",
                                    "Blocks Computed",
                                    "Total completed blocks emitted from compute submit.",
                                    "stelline-metrics-global-number",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getComputedBlocks() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("blocksLost",
                                    "Blocks Lost",
                                    "Total stale or evicted blocks.",
                                    "stelline-metrics-global-number",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getLostBlocks() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("blocksEmitted",
                                    "Blocks Emitted",
                                    "Total blocks successfully output from the module.",
                                    "stelline-metrics",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getEmittedBlocks() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("packetsReceived",
                                    "Packets Received",
                                    "Total received packets accepted into blocks.",
                                    "stelline-metrics",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getReceivedPackets() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("packetsEvicted",
                                    "Packets Evicted",
                                    "Packets discarded by offset or cutoff filtering.",
                                    "stelline-metrics",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getEvictedPackets() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("packetsLost",
                                    "Packets Lost",
                                    "Packets dropped because no block could be allocated.",
                                    "stelline-metrics",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getLostPackets() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("idleQueue",
                                    "Idle Queue",
                                    "Current idle queue depth.",
                                    "stelline-metrics",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getIdleQueue() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("receiveQueue",
                                    "Receive Queue",
                                    "Current receive queue depth.",
                                    "stelline-metrics",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getReceiveQueue() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("computeQueue",
                                    "Compute Queue",
                                    "Current compute queue depth.",
                                    "stelline-metrics",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getComputeQueue() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("burstsInFlight",
                                    "Bursts In Flight",
                                    "Current number of in-flight bursts.",
                                    "stelline-metrics-global-number",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getBurstsInFlight() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("avgBurstReleaseTimeUs",
                                    "Burst Release Time",
                                    "Average burst release time in microseconds.",
                                    "stelline-metrics",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getAverageBurstReleaseTimeUs() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("memPoolAvailable",
                                    "Memory Pool Available",
                                    "Current reusable output tensor pool availability.",
                                    "stelline-metrics",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getMemPoolAvailable() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("memPoolReferenced",
                                    "Memory Pool Referenced",
                                    "Current reusable output tensor pool references.",
                                    "stelline-metrics",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getMemPoolReferenced() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("blockMapLatestTimeIndex",
                                    "Block Map Latest Time Index",
                                    "Latest known block time index.",
                                    "stelline-metrics",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getBlockMapLatestTimeIndex() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("blockMapUsed",
                                    "Block Map Used",
                                    "Current number of active block map entries.",
                                    "stelline-metrics",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getBlockMapUsed() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("blockMapCapacity",
                                    "Block Map Capacity",
                                    "Maximum number of concurrent block map entries.",
                                    "stelline-metrics",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getBlockMapCapacity() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("payloadSizes",
                                    "Payload Sizes",
                                    "Observed payload sizes.",
                                    "stelline-metrics",
        [this]() -> std::any {
            return std::any(moduleImpl ? moduleImpl->getPayloadSizes() : std::string("[]"));
        }));

    JST_CHECK(defineInterfaceMetric("allAntennas",
                                    "All Antennas",
                                    "All observed antenna identifiers.",
                                    "stelline-metrics-global-string",
        [this]() -> std::any {
            return std::any(moduleImpl ? moduleImpl->getAllAntennas() : std::string("[]"));
        }));

    JST_CHECK(defineInterfaceMetric("filteredAntennas",
                                    "Filtered Antennas",
                                    "Accepted antenna identifiers after filtering.",
                                    "stelline-metrics",
        [this]() -> std::any {
            return std::any(moduleImpl ? moduleImpl->getFilteredAntennas() : std::string("[]"));
        }));

    JST_CHECK(defineInterfaceMetric("allChannels",
                                    "All Channels",
                                    "All observed channel identifiers.",
                                    "stelline-metrics-global-string",
        [this]() -> std::any {
            return std::any(moduleImpl ? moduleImpl->getAllChannels() : std::string("[]"));
        }));

    JST_CHECK(defineInterfaceMetric("filteredChannels",
                                    "Filtered Channels",
                                    "Accepted channel identifiers after filtering.",
                                    "stelline-metrics",
        [this]() -> std::any {
            return std::any(moduleImpl ? moduleImpl->getFilteredChannels() : std::string("[]"));
        }));

    JST_CHECK(defineInterfaceMetric("latestTimestamp",
                                    "Latest Timestamp",
                                    "Latest accepted packet timestamp.",
                                    "stelline-metrics-global-number",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getLatestTimestamp() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("burstsUsageDisplay",
                                    "Bursts Usage",
                                    "Burst utilization.",
                                    "progressbar",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::pair<std::string, F32>{"0/0", 0.0f};
            }
            const U64 inFlight = moduleImpl->getBurstsInFlight();
            const U64 capacity = moduleImpl->getMaxConcurrentBursts();
            const F32 value = capacity > 0 ? static_cast<F32>(inFlight) / static_cast<F32>(capacity) : 0.0f;
            const std::string label = jst::fmt::format("{}/{}", inFlight, capacity);
            return std::pair<std::string, F32>{label, value};
        }));

    JST_CHECK(defineInterfaceMetric("packetsReceivedDisplay",
                                    "Packets Received",
                                    "Total received packets.",
                                    "label",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getReceivedPackets() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("tensorsUsageDisplay",
                                    "Tensors Usage",
                                    "Tensor queue utilization (Receive, Compute, Swap).",
                                    "progressbar",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::pair<std::string, F32>{"0R:0C/0", 0.0f};
            }
            const U64 receive = moduleImpl->getReceiveQueue();
            const U64 compute = moduleImpl->getComputeQueue();
            const U64 swap = moduleImpl->getIdleQueue();
            const U64 total = receive + compute + swap;
            const F32 value = total > 0 ? static_cast<F32>(receive + compute) / static_cast<F32>(total) : 0.0f;
            const std::string label = jst::fmt::format("{}R:{}C/{}", receive, compute, total);
            return std::pair<std::string, F32>{label, value};
        }));

    JST_CHECK(defineInterfaceMetric("blocksReceivedProgressDisplay",
                                    "Blocks Received",
                                    "Received vs lost blocks progress.",
                                    "progressbar",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::pair<std::string, F32>{"0/0", 0.0f};
            }
            const U64 received = moduleImpl->getReceivedBlocks();
            const U64 lost = moduleImpl->getLostBlocks();
            const U64 total = received + lost;
            const F32 value = total > 0 ? static_cast<F32>(received) / static_cast<F32>(total) : 0.0f;
            const std::string label = jst::fmt::format("{}/{}", received, lost);
            return std::pair<std::string, F32>{label, value};
        }));

    JST_CHECK(defineInterfaceMetric("latestTimestampDisplay",
                                    "Latest Timestamp",
                                    "Latest accepted packet timestamp.",
                                    "label",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getLatestTimestamp() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("throughput",
                                    "Throughput",
                                    "Average observed packet payload throughput since module start.",
                                    "label",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{:.2f} Gbps", moduleImpl ? moduleImpl->getInputGbps() : 0.0));
        }));

    return Result::SUCCESS;
}

Result AtaReceiverImpl::create() {
    JST_CHECK(moduleCreate("ata_receiver", moduleConfig, {}));
    JST_CHECK(moduleExposeOutput("output", {"ata_receiver", "output"}));

    moduleImpl = moduleHandle("ata_receiver")->getImpl<Modules::AtaReceiverImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(AtaReceiverImpl);

}  // namespace Jetstream::Blocks
