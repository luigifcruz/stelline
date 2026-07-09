#ifndef STELLINE_ATA_RECEIVER_DETAIL_KERNEL_HH
#define STELLINE_ATA_RECEIVER_DETAIL_KERNEL_HH

#include <cstdint>
#include <string>

#include <cuda_runtime.h>

namespace Jetstream::Modules {

struct AtaReceiverBlockGeometry {
    std::uint64_t numberOfAntennas = 0;
    std::uint64_t numberOfChannels = 0;
    std::uint64_t numberOfSamples = 0;
    std::uint64_t numberOfPolarizations = 0;
};

cudaError_t LaunchAtaScatterKernel(const void* const* inputs,
                                   void* const* outputs,
                                   std::uint64_t numberOfPackets,
                                   const AtaReceiverBlockGeometry& totalShape,
                                   const AtaReceiverBlockGeometry& partialShape,
                                   const std::string& outputType,
                                   cudaStream_t stream);

}  // namespace Jetstream::Modules

#endif  // STELLINE_ATA_RECEIVER_DETAIL_KERNEL_HH
