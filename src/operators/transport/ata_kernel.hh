#ifndef STELLINE_OPERATORS_TRANSPORT_ATA_KERNEL_HH
#define STELLINE_OPERATORS_TRANSPORT_ATA_KERNEL_HH

#include <stelline/common.hh>
#include <stelline/yaml/types/block_shape.hh>

namespace stelline::operators::transport {

cudaError_t LaunchKernel(void* output, void** input, uint64_t numberOfPackets,
                         BlockShape fullShape, BlockShape partialShape, BlockShape slots,
                         cudaStream_t stream);

}  // namespace stelline::operators::transport

#endif  // STELLINE_OPERATORS_TRANSPORT_ATA_KERNEL_HH
