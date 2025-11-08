#ifndef STELLINE_OPERATORS_TRANSPORT_BLOCK_HH
#define STELLINE_OPERATORS_TRANSPORT_BLOCK_HH

#include <advanced_network/common.h>

#include <stelline/yaml/types/block_shape.hh>

#include "ata_kernel.hh"

using namespace gxf;
using namespace holoscan;
using namespace holoscan::ops;
using namespace holoscan::advanced_network;

namespace stelline::operators::transport {

struct Block {
    Block(const uint64_t& packetsPerBlock)
         : packetsPerBlock(packetsPerBlock) {
        bursts.reserve(packetsPerBlock);

        STELLINE_CUDA_CHECK_THROW(cudaMallocHost(&gpuData, packetsPerBlock * sizeof(void*)), [&]{
            HOLOSCAN_LOG_ERROR("[TRANSPORT] Failed to allocate memory for data block.");
        });

        STELLINE_CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), [&]{
            HOLOSCAN_LOG_ERROR("[TRANSPORT] Failed to create stream for data block.");
        });
    }

    ~Block() {
        destroy();

        cudaFreeHost(gpuData);
        cudaStreamDestroy(stream);
    }

    constexpr const uint64_t& index() const {
        return _index;
    }

    constexpr const uint64_t& timestamp() const {
        return _timestamp;
    }

    constexpr const std::shared_ptr<Tensor>& outputTensor() const {
        return _outputTensor;
    }

    inline void create(const uint64_t& index, const uint64_t& timestamp) {
        _index = index;
        _timestamp = timestamp;
    }

    inline void destroy() {
        _outputTensor.reset();
        packetCount = 0;
        bursts.clear();
    }

    inline void addPacket(const uint64_t& blockPacketIndex,
                          const uint64_t& burstPacketIndex,
                          const std::shared_ptr<BurstParams>& burst) {
        gpuData[blockPacketIndex] = get_segment_packet_ptr(burst.get(), RX_DATA, burstPacketIndex);
        bursts.insert(burst);
        packetCount += 1;
    }

    inline bool isComplete() const {
        return packetCount == packetsPerBlock;
    }

    inline bool isProcessing() const {
        return cudaStreamQuery(stream) != cudaSuccess;
    }

    inline void compute(std::shared_ptr<Tensor>& outputTensor,
                        const BlockShape& total,
                        const BlockShape& partial,
                        const BlockShape& slots) {
        _outputTensor = outputTensor;
        STELLINE_CUDA_CHECK_THROW(LaunchKernel(_outputTensor->data(), gpuData, packetsPerBlock,
                                               total, partial, slots,
                                               stream), [&]{
            HOLOSCAN_LOG_ERROR("[TRANSPORT] Failed to launch kernel for data block.");
        });
    }

 private:
    uint64_t _index;
    uint64_t _timestamp;
    uint64_t packetsPerBlock;

    void** gpuData;
    uint64_t packetCount;
    std::unordered_set<std::shared_ptr<BurstParams>> bursts;

    cudaStream_t stream;
    std::shared_ptr<Tensor> _outputTensor;
};

}  // namespace stelline::operators::transport

#endif  // STELLINE_OPERATORS_TRANSPORT_BLOCK_HH
