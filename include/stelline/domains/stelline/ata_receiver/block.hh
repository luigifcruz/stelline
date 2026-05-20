#ifndef STELLINE_DOMAINS_ATA_RECEIVER_BLOCK_HH
#define STELLINE_DOMAINS_ATA_RECEIVER_BLOCK_HH

#include <string>
#include <vector>

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct AtaReceiver : public Block::Config {
    std::string interfaceAddress;
    U64 gpuDeviceId = 0;
    U64 masterCore = 0;
    std::vector<U64> workerCores = {};
    std::string subscriptions;
    std::vector<U64> totalBlock = {1, 1, 1024, 1};
    std::vector<U64> partialBlock = {1, 1, 1024, 1};
    std::vector<U64> offsetBlock = {0, 0, 0, 0};
    std::string dataType = "CF32";
    U64 packetsPerBurst = 0;
    U64 maxConcurrentBursts = 0;
    U64 maxConcurrentBlocks = 4;
    U64 outputPoolSize = 2;

    JST_BLOCK_TYPE(ata_receiver);
    JST_BLOCK_DOMAIN("Stelline");
    JST_BLOCK_PARAMS(interfaceAddress, gpuDeviceId, masterCore, workerCores,
                     subscriptions, totalBlock, partialBlock, offsetBlock, dataType,
                     packetsPerBurst, maxConcurrentBursts, maxConcurrentBlocks,
                     outputPoolSize);
    JST_BLOCK_DESCRIPTION(
        "ATA Receiver",
        "Receives ATA voltage packets and assembles output tensors.",
        "// TODO: Write description."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_DOMAINS_ATA_RECEIVER_BLOCK_HH
