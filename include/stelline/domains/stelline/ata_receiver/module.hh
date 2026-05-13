#ifndef STELLINE_DOMAINS_ATA_RECEIVER_MODULE_HH
#define STELLINE_DOMAINS_ATA_RECEIVER_MODULE_HH

#include <string>
#include <vector>

#include <jetstream/module.hh>

namespace Jetstream::Modules {

struct AtaReceiver : public Module::Config {
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

    JST_MODULE_TYPE(ata_receiver);
    JST_MODULE_PARAMS(interfaceAddress, gpuDeviceId, masterCore, workerCores,
                      subscriptions, totalBlock, partialBlock, offsetBlock, dataType,
                      packetsPerBurst, maxConcurrentBursts, maxConcurrentBlocks,
                      outputPoolSize);
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_DOMAINS_ATA_RECEIVER_MODULE_HH
