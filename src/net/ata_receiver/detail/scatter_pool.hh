#ifndef STELLINE_ATA_RECEIVER_DETAIL_SCATTER_POOL_HH
#define STELLINE_ATA_RECEIVER_DETAIL_SCATTER_POOL_HH

#include <memory>
#include <queue>
#include <vector>

#include <cuda_runtime.h>

#include <daqiri/daqiri.h>

#include <jetstream/types.hh>

namespace Jetstream::Modules {

// Not thread-safe: owned by the packet processing thread, except
// create/destroy/drain which run with that thread stopped.
class ScatterStagingPool {
 public:
    struct Staging {
        const void** sources = nullptr;
        void** destinations = nullptr;
        cudaEvent_t event = nullptr;
    };

    Result create(const U64 maxConcurrentBursts, const U64 packetsPerBurst);
    Result destroy();

    Result acquire(Staging*& staging);
    void release(Staging* staging);
    Result submit(Staging* staging, const std::shared_ptr<daqiri::BurstParams>& burst);
    Result reap(const bool waitForOne);
    void drain();

    cudaStream_t stream() const {
        return scatterStream;
    }

 private:
    struct InFlight {
        Staging* staging = nullptr;
        std::shared_ptr<daqiri::BurstParams> burst;
    };

    cudaStream_t scatterStream = nullptr;
    std::vector<Staging> stagingPool;
    std::vector<Staging*> stagingFreeList;
    std::queue<InFlight> inFlightScatters;
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_ATA_RECEIVER_DETAIL_SCATTER_POOL_HH
