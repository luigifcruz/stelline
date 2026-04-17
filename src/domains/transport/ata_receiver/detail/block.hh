#ifndef STELLINE_DOMAINS_TRANSPORT_ATA_RECEIVER_BLOCK_HH
#define STELLINE_DOMAINS_TRANSPORT_ATA_RECEIVER_BLOCK_HH

#include <memory>
#include <string>
#include <unordered_set>

#include <advanced_network/common.h>

#include <jetstream/memory/tensor.hh>
#include <jetstream/logger.hh>
#include <jetstream/types.hh>

#include <jetstream/backend/devices/cuda/helpers.hh>

#include "kernel.hh"

namespace Jetstream::Modules {

struct AtaReceiverBlock {
    explicit AtaReceiverBlock(const U64 packetsPerBlock) : packetsPerBlock(packetsPerBlock) {
        bursts.reserve(packetsPerBlock);

        JST_CUDA_CHECK_THROW(cudaMallocHost(&gpuData, packetsPerBlock * sizeof(void*)), [&] {
            JST_ERROR("[MODULE_ATA_RECEIVER_NATIVE_CUDA] Failed to allocate ATA gather packet pointers.");
        });

        int lowPriority = 0;
        int highPriority = 0;
        JST_CUDA_CHECK_THROW(cudaDeviceGetStreamPriorityRange(&lowPriority, &highPriority), [&] {
            cudaFreeHost(gpuData);
            gpuData = nullptr;
            JST_ERROR("[MODULE_ATA_RECEIVER_NATIVE_CUDA] Failed to query CUDA stream priority range.");
        });

        JST_CUDA_CHECK_THROW(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, highPriority), [&] {
            cudaFreeHost(gpuData);
            gpuData = nullptr;
            JST_ERROR("[MODULE_ATA_RECEIVER_NATIVE_CUDA] Failed to create ATA gather CUDA stream.");
        });
    }

    ~AtaReceiverBlock() {
        destroy();
        if (gpuData) {
            cudaFreeHost(gpuData);
        }
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }

    void create(const U64 newIndex, const U64 newTimestamp) {
        index = newIndex;
        timestamp = newTimestamp;
        packetCount = 0;
    }

    void destroy() {
        outputTensor.reset();
        packetCount = 0;
        bursts.clear();
    }

    void addPacket(const U64 blockPacketIndex,
                   const U64 burstPacketIndex,
                   const std::shared_ptr<holoscan::advanced_network::BurstParams>& burst) {
        gpuData[blockPacketIndex] = holoscan::advanced_network::get_segment_packet_ptr(burst.get(), 1, burstPacketIndex);
        bursts.insert(burst);
        packetCount++;
    }

    bool isComplete() const {
        return packetCount == packetsPerBlock;
    }

    bool isProcessing() const {
        return cudaStreamQuery(stream) != cudaSuccess;
    }

    Result compute(const std::shared_ptr<Tensor>& newOutputTensor,
                   const AtaReceiverBlockGeometry& totalGeometry,
                   const AtaReceiverBlockGeometry& partialGeometry,
                   const AtaReceiverBlockGeometry& slotGeometry,
                   const std::string& outputType) {
        outputTensor = newOutputTensor;

        JST_CUDA_CHECK(LaunchAtaGatherKernel(outputTensor->data(),
                                             gpuData,
                                             packetsPerBlock,
                                             totalGeometry,
                                             partialGeometry,
                                             slotGeometry,
                                             outputType,
                                             stream), [&] {
            outputTensor.reset();
        });

        return Result::SUCCESS;
    }

    U64 index = 0;
    U64 timestamp = 0;
    U64 packetsPerBlock = 0;
    U64 packetCount = 0;
    void** gpuData = nullptr;
    cudaStream_t stream = nullptr;
    std::unordered_set<std::shared_ptr<holoscan::advanced_network::BurstParams>> bursts;
    std::shared_ptr<Tensor> outputTensor;
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_DOMAINS_TRANSPORT_ATA_RECEIVER_BLOCK_HH
