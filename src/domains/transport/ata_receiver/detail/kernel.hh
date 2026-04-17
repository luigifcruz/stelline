#ifndef STELLINE_DOMAINS_TRANSPORT_ATA_RECEIVER_KERNEL_HH
#define STELLINE_DOMAINS_TRANSPORT_ATA_RECEIVER_KERNEL_HH

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

cudaError_t LaunchAtaGatherKernel(void* output,
                                  void** input,
                                  std::uint64_t numberOfPackets,
                                  const AtaReceiverBlockGeometry& totalShape,
                                  const AtaReceiverBlockGeometry& partialShape,
                                  const AtaReceiverBlockGeometry& slotShape,
                                  const std::string& outputType,
                                  cudaStream_t stream);

}  // namespace Jetstream::Modules

#endif  // STELLINE_DOMAINS_TRANSPORT_ATA_RECEIVER_KERNEL_HH
