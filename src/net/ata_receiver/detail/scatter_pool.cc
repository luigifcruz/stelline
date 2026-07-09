#include "scatter_pool.hh"

#include <jetstream/logger.hh>
#include <jetstream/macros.hh>
#include <jetstream/backend/devices/cuda/helpers.hh>

namespace Jetstream::Modules {

Result ScatterStagingPool::create(const U64 maxConcurrentBursts, const U64 packetsPerBurst) {
    int lowPriority = 0;
    int highPriority = 0;
    JST_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lowPriority, &highPriority), [&] {
        JST_ERROR("[MODULE_ATA_RECEIVER_NATIVE_CUDA] Failed to query CUDA stream priority range: {}", err);
    });
    JST_CUDA_CHECK(cudaStreamCreateWithPriority(&scatterStream, cudaStreamNonBlocking, highPriority), [&] {
        JST_ERROR("[MODULE_ATA_RECEIVER_NATIVE_CUDA] Failed to create ATA scatter stream: {}", err);
    });

    stagingPool.resize(maxConcurrentBursts);
    for (auto& staging : stagingPool) {
        JST_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&staging.sources),
                                      packetsPerBurst * sizeof(void*)), [&] {
            JST_ERROR("[MODULE_ATA_RECEIVER_NATIVE_CUDA] Failed to allocate scatter staging sources: {}", err);
        });
        JST_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&staging.destinations),
                                      packetsPerBurst * sizeof(void*)), [&] {
            JST_ERROR("[MODULE_ATA_RECEIVER_NATIVE_CUDA] Failed to allocate scatter staging destinations: {}", err);
        });
        JST_CUDA_CHECK(cudaEventCreateWithFlags(&staging.event, cudaEventDisableTiming), [&] {
            JST_ERROR("[MODULE_ATA_RECEIVER_NATIVE_CUDA] Failed to create scatter staging event: {}", err);
        });
        stagingFreeList.push_back(&staging);
    }

    return Result::SUCCESS;
}

Result ScatterStagingPool::destroy() {
    drain();

    for (auto& staging : stagingPool) {
        if (staging.event) {
            cudaEventDestroy(staging.event);
        }
        if (staging.sources) {
            cudaFreeHost(staging.sources);
        }
        if (staging.destinations) {
            cudaFreeHost(staging.destinations);
        }
    }
    stagingPool.clear();
    stagingFreeList.clear();

    if (scatterStream) {
        cudaStreamDestroy(scatterStream);
        scatterStream = nullptr;
    }

    return Result::SUCCESS;
}

Result ScatterStagingPool::acquire(Staging*& staging) {
    if (stagingFreeList.empty()) {
        JST_CHECK(reap(true));
    }
    if (stagingFreeList.empty()) {
        JST_ERROR("[MODULE_ATA_RECEIVER_NATIVE_CUDA] No scatter staging buffer available.");
        return Result::ERROR;
    }

    staging = stagingFreeList.back();
    stagingFreeList.pop_back();

    return Result::SUCCESS;
}

void ScatterStagingPool::release(Staging* staging) {
    stagingFreeList.push_back(staging);
}

Result ScatterStagingPool::submit(Staging* staging, const std::shared_ptr<daqiri::BurstParams>& burst) {
    JST_CUDA_CHECK(cudaEventRecord(staging->event, scatterStream), [&] {
        JST_ERROR("[MODULE_ATA_RECEIVER_NATIVE_CUDA] Failed to record scatter staging event: {}", err);
    });
    inFlightScatters.push({staging, burst});

    return Result::SUCCESS;
}

Result ScatterStagingPool::reap(const bool waitForOne) {
    while (!inFlightScatters.empty()) {
        auto& entry = inFlightScatters.front();

        auto status = cudaEventQuery(entry.staging->event);
        if (status == cudaErrorNotReady) {
            if (!(waitForOne && stagingFreeList.empty())) {
                break;
            }
            status = cudaEventSynchronize(entry.staging->event);
        }
        if (status != cudaSuccess) {
            JST_ERROR("[MODULE_ATA_RECEIVER_NATIVE_CUDA] Scatter kernel failed: {}",
                      cudaGetErrorString(status));
            return Result::ERROR;
        }

        stagingFreeList.push_back(entry.staging);
        inFlightScatters.pop();
    }

    return Result::SUCCESS;
}

void ScatterStagingPool::drain() {
    if (scatterStream) {
        cudaStreamSynchronize(scatterStream);
    }

    while (!inFlightScatters.empty()) {
        stagingFreeList.push_back(inFlightScatters.front().staging);
        inFlightScatters.pop();
    }
}

}  // namespace Jetstream::Modules
