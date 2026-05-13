#ifndef STELLINE_DOMAINS_ATA_RECEIVER_MODULE_IMPL_HH
#define STELLINE_DOMAINS_ATA_RECEIVER_MODULE_IMPL_HH

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <stelline/domains/stelline/ata_receiver/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/memory/tensor.hh>
#include <jetstream/tools/snapshot.hh>
#include <advanced_network/common.h>

namespace ano = holoscan::advanced_network;

namespace Jetstream::Modules {

struct AtaReceiverBlock;

struct AtaReceiverReadyTensor {
    std::shared_ptr<Tensor> tensor;
    U64 timestamp = 0;
};

constexpr U64 kShapeRank = 4;
constexpr U64 kAntennaAxis = 0;
constexpr U64 kChannelAxis = 1;
constexpr U64 kSampleAxis = 2;
constexpr U64 kPolarizationAxis = 3;

constexpr U64 kPacketDataSize = 6144;
constexpr U64 kPacketHeaderSize = 16;
constexpr U64 kPacketHeaderOffset = 42;

constexpr int kRxHeaderSegment = 0;
constexpr int kRxDataSegment = 1;

struct AtaReceiverImpl : public Module::Impl, public DynamicConfig<AtaReceiver> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

    U64 getReceivedBlocks() const;
    U64 getComputedBlocks() const;
    U64 getLostBlocks() const;
    U64 getReceivedPackets() const;
    U64 getEvictedPackets() const;
    U64 getLostPackets() const;
    U64 getBurstsInFlight() const;
    U64 getMemPoolAvailable() const;
    U64 getMemPoolReferenced() const;
    U64 getBlockMapLatestTimeIndex() const;
    U64 getBlockMapUsed() const;
    U64 getBlockMapCapacity() const;
    U64 getLatestTimestamp() const;
    F64 getInputGbps() const;
    U64 getIdleQueue() const;
    U64 getReceiveQueue() const;
    U64 getComputeQueue() const;
    U64 getAverageBurstReleaseTimeUs() const;
    U64 getEmittedBlocks() const;
    U64 getMaxConcurrentBursts() const;
    std::string getPayloadSizes() const;
    std::string getAllAntennas() const;
    std::string getFilteredAntennas() const;
    std::string getAllChannels() const;
    std::string getFilteredChannels() const;

 protected:
     Tensor outputTensor;

     std::vector<U64> slotShape;
     U64 packetsPerBlock = 0;
     U64 packetDuration = 0;
     U64 blockDuration = 0;
     U64 timestampCutoff = 0;

     std::atomic<bool> errored = false;
     std::atomic<bool> packetProcessingThreadRunning = false;
     std::atomic<bool> burstCollectorThreadRunning = false;
     std::thread packetProcessingThread;
     std::thread burstCollectorThread;

     std::queue<std::shared_ptr<AtaReceiverBlock>> idleQueue;
     std::queue<std::shared_ptr<AtaReceiverBlock>> receiveQueue;
     std::queue<std::shared_ptr<AtaReceiverBlock>> computeQueue;
     std::queue<std::shared_ptr<AtaReceiverBlock>> swapQueue;
     std::unordered_map<U64, std::shared_ptr<AtaReceiverBlock>> blockMap;

     std::mutex poolMutex;
     std::condition_variable outputPoolCv;
     std::queue<std::shared_ptr<Tensor>> availableOutputTensors;

     std::mutex readyMutex;
     std::queue<AtaReceiverReadyTensor> readyOutputTensors;

     std::mutex burstCollectorMutex;
     std::unordered_set<std::shared_ptr<ano::BurstParams>> bursts;

     Tools::Snapshot<U64> latestBlockTimeIndex{0};
     Tools::Snapshot<U64> latestTimestamp{0};
     Tools::Snapshot<U64> blockMapDepth{0};

     Tools::Snapshot<U64> receivedPackets{0};
     Tools::Snapshot<U64> evictedPackets{0};
     Tools::Snapshot<U64> lostPackets{0};
     Tools::Snapshot<U64> receivedBlocks{0};
     Tools::Snapshot<U64> emittedBlocks{0};
     Tools::Snapshot<U64> lostBlocks{0};

     Tools::Snapshot<U64> idleQueueDepth{0};
     Tools::Snapshot<U64> receiveQueueDepth{0};
     Tools::Snapshot<U64> computeQueueDepth{0};
     Tools::Snapshot<U64> readyQueueDepth{0};
     Tools::Snapshot<U64> outputPoolDepth{0};
     Tools::Snapshot<U64> burstsInFlight{0};
     Tools::Snapshot<U64> avgBurstReleaseTimeUs{0};
     Tools::Snapshot<F64> throughputGbps{0.0};

     Tools::Snapshot<std::set<U64>> allAntennas;
     Tools::Snapshot<std::set<U64>> filteredAntennas;
     Tools::Snapshot<std::set<U64>> allChannels;
     Tools::Snapshot<std::set<U64>> filteredChannels;
     Tools::Snapshot<std::set<U64>> payloadSizes;
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_DOMAINS_ATA_RECEIVER_MODULE_IMPL_HH
