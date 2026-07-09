#ifndef STELLINE_ATA_RECEIVER_DETAIL_DAQIRI_CONFIG_HH
#define STELLINE_ATA_RECEIVER_DETAIL_DAQIRI_CONFIG_HH

#include <string>
#include <vector>

#include <daqiri/daqiri.h>

#include <jetstream/types.hh>

#include "../../endpoint.hh"

namespace Jetstream::Modules {

struct DaqiriRxConfigParams {
    std::string interfaceAddress;
    U64 gpuDeviceId = 0;
    U64 masterCore = 0;
    std::vector<U64> workerCores;
    U64 packetsPerBurst = 0;
    U64 maxConcurrentBursts = 0;
    daqiri::MemoryKind dataMemoryKind = daqiri::MemoryKind::DEVICE;
};

Result BuildDaqiriRxConfig(const DaqiriRxConfigParams& params,
                           const std::vector<stelline::domains::stelline::utils::SubscriptionEndpoint>& subscriptions,
                           daqiri::NetworkConfig& cfg);

}  // namespace Jetstream::Modules

#endif  // STELLINE_ATA_RECEIVER_DETAIL_DAQIRI_CONFIG_HH
