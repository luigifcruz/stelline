#ifndef STELLINE_ATA_RECEIVER_DETAIL_BLOCK_HH
#define STELLINE_ATA_RECEIVER_DETAIL_BLOCK_HH

#include <algorithm>
#include <memory>
#include <vector>

#include <jetstream/memory/tensor.hh>
#include <jetstream/logger.hh>
#include <jetstream/types.hh>

#include <jetstream/backend/devices/cuda/helpers.hh>

namespace Jetstream::Modules {

struct AtaReceiverBlock {
    explicit AtaReceiverBlock(const U64 packetsPerBlock)
        : packetsPerBlock(packetsPerBlock),
          slotFilled(packetsPerBlock, 0) {
        JST_CUDA_CHECK_THROW(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), [&] {
            JST_ERROR("[MODULE_ATA_RECEIVER_NATIVE_CUDA] Failed to create ATA block event.");
        });
    }

    ~AtaReceiverBlock() {
        destroy();
        if (event) {
            cudaEventDestroy(event);
        }
    }

    void create(const U64 newIndex,
                const U64 newTimestamp,
                const std::shared_ptr<Tensor>& tensor) {
        index = newIndex;
        timestamp = newTimestamp;
        packetCount = 0;
        emit = false;
        outputTensor = tensor;
        std::fill(slotFilled.begin(), slotFilled.end(), 0);
    }

    void destroy() {
        outputTensor.reset();
        packetCount = 0;
        emit = false;
    }

    bool claimSlot(const U64 blockPacketIndex) {
        if (slotFilled[blockPacketIndex]) {
            return false;
        }
        slotFilled[blockPacketIndex] = 1;
        packetCount++;
        return true;
    }

    bool isComplete() const {
        return packetCount == packetsPerBlock;
    }

    bool isDone() const {
        return cudaEventQuery(event) == cudaSuccess;
    }

    bool isFailed() const {
        const auto status = cudaEventQuery(event);
        return status != cudaSuccess && status != cudaErrorNotReady;
    }

    U64 index = 0;
    U64 timestamp = 0;
    U64 packetsPerBlock = 0;
    U64 packetCount = 0;
    bool emit = false;
    cudaEvent_t event = nullptr;
    std::vector<U8> slotFilled;
    std::shared_ptr<Tensor> outputTensor;
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_ATA_RECEIVER_DETAIL_BLOCK_HH
